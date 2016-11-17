package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.Batch._
import cats._

import scala.language.existentials
import scala.language.implicitConversions
import scala.language.higherKinds
import cats.implicits._

import scala.annotation.elidable

object NeuralNetwork /*extends LowPriortyDifferentiableFunction*/ {

  /** @template */
  type Aux[-Input0 <: Batch, +Output0 <: Batch] =
    NeuralNetwork {
      type Input >: Input0
      type Output <: Output0
    }

  // TODO: Rename this trait to Layer and move to a separate library
  trait Cached extends NeuralNetwork {

    private[deepLearning] val cache =
      java.util.Collections.synchronizedMap(new java.util.IdentityHashMap[BatchId.Aux[Input], SharedBatch](1))

    protected trait ReferenceCount extends BatchId with Batch { this: SharedBatch =>

      /**
        * Returns a wrapped [[Batch]] able to detect error of closing more than once if ASSERTION is enabled,
        * or returns this [[ReferenceCount]] itself when ASSERTION is disabled hence no check.
        */
      override final def open(): Open = {
        // TODO: reopen
        assert(!closingFlag.closed)
        val newCount = synchronized {
          val newCount = count + 1
          count = newCount
          newCount
        }
        assert(newCount >= 1)
        if (newCount == 1) {
          cache.put(input, this)
        }

        // Returns a [[Batch]] able to detect error of closing more than once.
        @elidable(elidable.ASSERTION)
        def checkIfCloseOnlyOnce = new Batch {
          type Delta = ReferenceCount.this.Delta
          type Data = ReferenceCount.this.Data

          override final def backward(delta: Delta) = ReferenceCount.this.backward(delta)

          def value = ReferenceCount.this.value

          private var closed = false

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

        Option(checkIfCloseOnlyOnce).getOrElse(ReferenceCount.this.self)
      }

      override type Open >: Batch.Aux[Data, Delta] <: Batch.Aux[Data, Delta]

      private final def self: Open = this: Batch.Aux[Data, Delta]

      private[Cached] var count: Int = 0

      protected def flush(): Unit

      /**
        * Closes upstream batch of this [[SharedBatch]]
        */
      protected def closeUpstreams(): Unit

      def input: BatchId.Aux[Input]

      private[Cached] final class ClosingFlag {
        var closed = false
        @elidable(elidable.ASSERTION)
        def assertNotClosed() = {
          assert(!closed)
          closed = true
        }
      }

      @elidable(elidable.ASSERTION)
      private val closingFlag = new ClosingFlag

      /**
        * FIXME: rename to release
        */
      override final def close(): Unit = {
        val newCount = synchronized {
          val newCount = count - 1
          count = newCount
          newCount
        }
        assert(newCount >= 0)
        if (newCount == 0) {
          closingFlag.assertNotClosed()
          val batch: SharedBatch = cache.remove(input)
          assert(batch eq (this: SharedBatch))
          flush()
          closeUpstreams()
        }
      }

    }

    protected trait MonoidBatch extends ReferenceCount { this: SharedBatch =>

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

      override final def backward(delta: Delta): Unit = {
        synchronized {
          currentDelta = currentDelta |+| delta
        }
      }
    }

    protected trait SemigroupBatch extends ReferenceCount { this: SharedBatch =>

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

      override final def backward(delta: Delta): Unit = {
        synchronized {
          currentDelta = currentDelta |+| Some(delta)
        }
      }
    }

    type Output = SharedBatch#Open

    protected type SharedBatch <: ReferenceCount

    /**
      * Performs the underlying forward pass.
      *
      * @return a [[Batch]] that will be cached for subsequent [[#forward]]
      */
    protected def rawForward(input: BatchId.Aux[Input]): SharedBatch

    override final def forward(input: BatchId.Aux[Input]): BatchId.Aux[Output] = {
      cache.get(input) match {
        case null =>
          rawForward(input)
        case sharedBatch =>
          sharedBatch
      }
    }
  }

}

trait NeuralNetwork {

  import NeuralNetwork._

  type Input <: Batch

  type Output <: Batch

  def forward(input: BatchId.Aux[Input]): BatchId.Aux[Output]

}
