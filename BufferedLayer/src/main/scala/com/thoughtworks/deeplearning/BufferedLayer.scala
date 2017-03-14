package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Tape
import cats._
import cats.implicits._

import annotation.elidable

// TODO: Review if the reference count works correctly
/**
  * BufferedLayer records whether the parameters in the Layer are in use by reference counting,
  * in the `forward ()` ,parameters will be cached to `cache`,
  * If a parameter is no longer used (ie `value` is `0`) will call `close ()` to release the resource (ie, remove the cached data from `cache`)
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait BufferedLayer extends Layer {

  private[deeplearning] val cache =
    java.util.Collections.synchronizedMap(new java.util.IdentityHashMap[AnyRef, BufferedTape](1))

  protected trait ReferenceCount extends Tape { this: BufferedTape =>

    // Returns a [[Tape]] able to detect error of closing more than once.
    @elidable(elidable.ASSERTION)
    private def checked = new Tape {
      override type Delta = ReferenceCount.this.Delta
      override type Data = ReferenceCount.this.Data

      override final def addReference() = ReferenceCount.this.addReference()

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

    private[BufferedLayer] final def checkedIfCloseOnlyOnce: Self = {
      Option(checked).getOrElse(ReferenceCount.this.self)
    }

    /**
      * Returns a wrapped [[com.thoughtworks.deeplearning.Layer.Tape Tape]] able to detect error of closing more than once if ASSERTION is enabled,
      * or returns this [[ReferenceCount]] itself when ASSERTION is disabled hence no check.
      */
    override final def addReference(): Self = {
      val newCount = synchronized {
        val newCount = count + 1
        count = newCount
        newCount
      }
      assert(newCount >= 1)
      checkedIfCloseOnlyOnce
    }

    private[BufferedLayer] type Self >: Tape.Aux[Data, Delta] <: Tape.Aux[Data, Delta]

    private final def self: Self = this: Tape.Aux[Data, Delta]

    private[BufferedLayer] var count: Int = 1

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
        val tape: BufferedTape = cache.remove(input)
        assert(tape eq (this: BufferedTape))
        flush()
        closeUpstreams()
      }
    }

  }

  protected trait MonoidTape extends ReferenceCount { this: BufferedTape =>

    private var currentDelta: Delta = monoid.empty

    /**
      * Performs the underlying backward pass with all `upstreamDelta`s that previously received from [[forceBackward]].
      */
    protected def rawBackward(delta: Delta): Unit

    implicit protected def monoid: Monoid[Delta]

    override protected final def flush(): Unit = {
      rawBackward(synchronized {
        val delta = currentDelta
        currentDelta = monoid.empty
        delta
      })
    }

    override final protected def forceBackward(delta: Delta): Unit = {
      synchronized {
        currentDelta = currentDelta |+| delta
      }
    }
  }

  protected trait SemigroupTape extends ReferenceCount { this: BufferedTape =>

    private var currentDelta: Option[Delta] = None

    protected def rawBackward(delta: Delta): Unit

    implicit protected def semigroup: Semigroup[Delta]

    override protected final def flush(): Unit = {
      synchronized {
        val delta = currentDelta
        currentDelta = None
        delta
      }.foreach(rawBackward)
    }

    override final protected def forceBackward(delta: Delta): Unit = {
      synchronized {
        currentDelta |+|= Some(delta)
      }
    }
  }

  /** @template */
  type Output = BufferedTape#Self

  protected type BufferedTape <: ReferenceCount

  /**
    * Performs the underlying forward pass.
    *
    * @return a [[com.thoughtworks.deeplearning.Layer.Tape Tape]] that will be cached for subsequent [[forward]]
    */
  protected def rawForward(input: Input): BufferedTape

  override final def forward(input: Input): Output = {
    cache.get(input) match {
      case null =>
        val savedInput = input.addReference().asInstanceOf[Input] // FIXME: Add self type in Tape to avoid asInstanceOf
        val tape = rawForward(savedInput)
        cache.put(savedInput, tape).ensuring(_ == null)
        tape.checkedIfCloseOnlyOnce
      case sharedTape =>
        sharedTape.addReference()
    }
  }
}

object BufferedLayer {

  /**
    * A helper that contains common boilerplate code for layers of unary operator
    *
    * @example{{{
    * final case class UnaryOps[Input0 <: Tape](
    * operand: Layer.Aux[Input0, INDArrayPlaceholder.Tape]) extends BufferedLayer.Unary {}
    * }}}
    */
  trait Unary extends BufferedLayer {

    protected val operand: Layer.Aux[Input, _ <: Tape]

    protected type BufferedTape <: UnaryTape

    protected trait UnaryTape extends ReferenceCount { this: BufferedTape =>

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
    * A helper that contains common boilerplate code for layers of binary operator layer
    *
    * @example{{{
    * final case class BinaryOps[Input0 <: Tape](
    * operand1: Layer.Aux[Input0, INDArrayPlaceholder.Tape],
    * operand2: Layer.Aux[Input0, INDArrayPlaceholder.Tape]) extends BufferedLayer.Binary {}
    * }}}
    */
  trait Binary extends BufferedLayer {

    protected val operand1: Layer.Aux[Input, _ <: Tape]
    protected val operand2: Layer.Aux[Input, _ <: Tape]

    protected type BufferedTape <: BinaryTape

    protected trait BinaryTape extends ReferenceCount { this: BufferedTape =>

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
