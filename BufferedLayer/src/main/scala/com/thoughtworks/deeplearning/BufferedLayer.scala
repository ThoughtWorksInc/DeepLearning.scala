package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Batch
import cats._
import cats.implicits._

import annotation.elidable

// TODO: Review if the reference count works correctly
/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait BufferedLayer extends Layer {

  private[deeplearning] val cache =
    java.util.Collections.synchronizedMap(new java.util.IdentityHashMap[AnyRef, BufferedBatch](1))

  protected trait ReferenceCount extends Batch { this: BufferedBatch =>

    // Returns a [[Batch]] able to detect error of closing more than once.
    @elidable(elidable.ASSERTION)
    private def checked = new Batch {
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
      * Returns a wrapped [[com.thoughtworks.deeplearning.Layer.Batch Batch]] able to detect error of closing more than once if ASSERTION is enabled,
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

    private[BufferedLayer] type Self >: Batch.Aux[Data, Delta] <: Batch.Aux[Data, Delta]

    private final def self: Self = this: Batch.Aux[Data, Delta]

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
        val batch: BufferedBatch = cache.remove(input)
        assert(batch eq (this: BufferedBatch))
        flush()
        closeUpstreams()
      }
    }

  }

  protected trait MonoidBatch extends ReferenceCount { this: BufferedBatch =>

    private var currentDelta: Delta = monoid.empty

    /**
      * Performs the underlying backward pass with all `upstreamDelta`s that previously received from [[#backward]].
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

  protected trait SemigroupBatch extends ReferenceCount { this: BufferedBatch =>

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
  type Output = BufferedBatch#Self

  protected type BufferedBatch <: ReferenceCount

  /**
    * Performs the underlying forward pass.
    *
    * @return a [[com.thoughtworks.deeplearning.Layer.Batch Batch]] that will be cached for subsequent [[forward]]
    */
  protected def rawForward(input: Input): BufferedBatch

  override final def forward(input: Input): Output = {
    cache.get(input) match {
      case null =>
        val savedInput = input.addReference().asInstanceOf[Input] // FIXME: Add self type in Batch to avoid asInstanceOf
        val batch = rawForward(savedInput)
        cache.put(savedInput, batch).ensuring(_ == null)
        batch.checkedIfCloseOnlyOnce
      case sharedBatch =>
        sharedBatch.addReference()
    }
  }
}

object BufferedLayer {

  /**
    * A helper that contains common boilerplate code for layers of unary operator
    *
    * {{{
    * final case class UnaryOps[Input0 <: Batch](
    * operand: Layer.Aux[Input0, INDArrayPlaceholder.Batch]) extends BufferedLayer.Unary {}
    * }}}
    */
  trait Unary extends BufferedLayer {

    protected val operand: Layer.Aux[Input, _ <: Batch]

    protected type BufferedBatch <: UnaryBatch

    protected trait UnaryBatch extends ReferenceCount { this: BufferedBatch =>

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
    * Implement a binary operator layer
    * {{{
    * final case class BinaryOps[Input0 <: Batch](
    * operand1: Layer.Aux[Input0, INDArrayPlaceholder.Batch],
    * operand2: Layer.Aux[Input0, INDArrayPlaceholder.Batch]) extends BufferedLayer.Binary {}
    * }}}
    */
  trait Binary extends BufferedLayer {

    protected val operand1: Layer.Aux[Input, _ <: Batch]
    protected val operand2: Layer.Aux[Input, _ <: Batch]

    protected type BufferedBatch <: BinaryBatch

    protected trait BinaryBatch extends ReferenceCount { this: BufferedBatch =>

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
