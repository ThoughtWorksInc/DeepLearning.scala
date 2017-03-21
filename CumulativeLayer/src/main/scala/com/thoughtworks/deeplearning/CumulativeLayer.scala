package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Tape
import cats._
import cats.implicits._

import annotation.elidable

/**
  * A [[Layer]] that minimizes the computation during both [[forward]] pass and [[Tape.backward backward]] pass.
  *
  * For [[forward]] pass, the result will be cached.
  *
  * For [[Tape.backward backward]] pass, the [[Tape]] is accumulated until [[Tape.flush flush]].
  *
  * @see [[Layer.Output]]
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait CumulativeLayer extends Layer {

  private[deeplearning] val cache =
    java.util.Collections.synchronizedMap(new java.util.IdentityHashMap[AnyRef, CumulativeTape](1))

  protected trait ReferenceCount extends Tape { this: CumulativeTape =>

    /**
      * Returns a [[Layer.Tape Tape]] that prevents [[Layer.Tape#close close]] being invoked more than once.
      */
    @elidable(elidable.ASSERTION)
    private def checked = new Tape {
      override type Delta = ReferenceCount.this.Delta
      override type Data = ReferenceCount.this.Data

      override final def duplicate() = ReferenceCount.this.duplicate()

      override final protected def forceBackward(delta: Delta) = ReferenceCount.this.forceBackward(delta)

      override final def isTrainable: Boolean = ReferenceCount.this.isTrainable

      override final def value = ReferenceCount.this.value

      private var closed = false

      override protected final def finalize(): Unit = {
        assert(closed)
      }

      override final def close(): Unit = {
        ReferenceCount.this.synchronized {
          if (closed) {
            throw new IllegalStateException("close() method must be called once and only once.")
          } else {
            closed = true
          }
        }
        ReferenceCount.this.close()
      }
    }

    private[CumulativeLayer] final def checkedIfCloseOnlyOnce: Self = {
      Option(checked).getOrElse(ReferenceCount.this.self)
    }

    /**
      * Returns a wrapped [[com.thoughtworks.deeplearning.Layer.Tape Tape]] able to detect error of closing more than once if ASSERTION is enabled,
      * or returns this [[ReferenceCount]] itself when ASSERTION is disabled hence no check.
      */
    override final def duplicate(): Self = {
      val newCount = synchronized {
        val newCount = count + 1
        count = newCount
        newCount
      }
      assert(newCount >= 1)
      checkedIfCloseOnlyOnce
    }

    private[CumulativeLayer] type Self >: Tape.Aux[Data, Delta] <: Tape.Aux[Data, Delta]

    private final def self: Self = this: Tape.Aux[Data, Delta]

    private[CumulativeLayer] var count: Int = 1

    protected def flush(): Unit

    protected def input: AnyRef

    protected def closeUpstreams(): Unit

    override final def close(): Unit = {
      val newCount = synchronized {
        val newCount = count - 1
        count = newCount
        newCount
      }
      assert(newCount >= 0)
      if (newCount == 0) {
        val tape: CumulativeTape = cache.remove(input)
        assert(tape eq (this: CumulativeTape))
        flush()
        closeUpstreams()
      }
    }

  }

  protected trait MonoidTape extends ReferenceCount { this: CumulativeTape =>

    private var accumulatedDelta: Delta = monoid.empty

    /**
      * Performs the underlying backward pass with the accumulated `outputDelta`s previously received from several [[forceBackward]] calls.
      */
    protected def rawBackward(accumulatedDelta: Delta): Unit

    implicit protected def monoid: Monoid[Delta]

    override protected final def flush(): Unit = {
      rawBackward(synchronized {
        val delta = accumulatedDelta
        accumulatedDelta = monoid.empty
        delta
      })
    }

    override protected final def forceBackward(outputDelta: Delta): Unit = {
      synchronized {
        accumulatedDelta |+|= outputDelta
      }
    }
  }

  protected trait SemigroupTape extends ReferenceCount { this: CumulativeTape =>

    private var accumulatedDelta: Option[Delta] = None

    protected def rawBackward(accumulatedDelta: Delta): Unit

    implicit protected def semigroup: Semigroup[Delta]

    override protected final def flush(): Unit = {
      synchronized {
        val delta = accumulatedDelta
        accumulatedDelta = None
        delta
      }.foreach(rawBackward)
    }

    override protected final def forceBackward(outputDelta: Delta): Unit = {
      synchronized {
        accumulatedDelta |+|= Some(outputDelta)
      }
    }
  }

  /**
    * A cumulative [[Tape]] returned by [[forward]].
    *
    * When this [[Output]] is [[Tape.backward backward]]ing,
    * the `delta` parameter will not be back-propagated to its upstreams immediately.
    * Instead, the `delta` parameter will be accumulated internally.
    * Then, when this [[Output]] is [[flush]]ing,
    * the delta accumulator will be processed and back-propagated to its upstreams.
    *
    * This [[Output]] is reference counted.
    * When the last instance of all this [[Output]]'s [[duplicate]]s is [[close]]d,
    * [[flush]] will be called and all the upstreams will be closed as well.
    *
    * @template */
  type Output = CumulativeTape#Self

  protected type CumulativeTape <: ReferenceCount

  /**
    * Performs the underlying [[forward]] pass.
    *
    * @return a [[com.thoughtworks.deeplearning.Layer.Tape Tape]] that will be cached for subsequent [[forward]]
    */
  protected def rawForward(input: Input): CumulativeTape

  /**
    * Returns the returns the result of [[rawForward]].
    *
    * If this method is called more than once with the same `input` parameter, during one iteration,
    * the result will be cached and the [[rawForward]] will be executed only once.
    */
  override final def forward(input: Input): Output = {
    cache.get(input) match {
      case null =>
        val savedInput = input.duplicate().asInstanceOf[Input] // FIXME: Add self type in Tape to avoid asInstanceOf
        val tape = rawForward(savedInput)
        cache.put(savedInput, tape).ensuring(_ == null)
        tape.checkedIfCloseOnlyOnce
      case sharedTape =>
        sharedTape.duplicate()
    }
  }
}

object CumulativeLayer {

  /**
    * A helper that contains common boilerplate code for layers of unary operator.
    */
  trait Unary extends CumulativeLayer {

    protected val operand: Layer.Aux[Input, _ <: Tape]

    protected type CumulativeTape <: UnaryTape

    protected trait UnaryTape extends ReferenceCount { this: CumulativeTape =>

      override def input: Input

      protected val upstream: operand.Output = operand.forward(input)

      override final val isTrainable: Boolean = upstream.isTrainable

      override protected final def closeUpstreams(): Unit = {
        upstream.close()
        input.close()
      }

    }

  }

  /**
    * A helper that contains common boilerplate code for layers of binary operator.
    */
  trait Binary extends CumulativeLayer {

    protected val operand1: Layer.Aux[Input, _ <: Tape]
    protected val operand2: Layer.Aux[Input, _ <: Tape]

    protected type CumulativeTape <: BinaryTape

    protected trait BinaryTape extends ReferenceCount { this: CumulativeTape =>

      override def input: Input

      protected val upstream1: operand1.Output = operand1.forward(input)
      protected val upstream2: operand2.Output = operand2.forward(input)

      override final val isTrainable: Boolean = upstream1.isTrainable || upstream2.isTrainable

      override protected final def closeUpstreams(): Unit = {
        upstream1.close()
        upstream2.close()
        input.close()
      }

    }

  }

}
