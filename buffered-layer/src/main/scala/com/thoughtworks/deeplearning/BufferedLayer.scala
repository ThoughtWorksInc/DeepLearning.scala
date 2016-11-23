package com.thoughtworks.deeplearning

import cats._
import cats.implicits._

import scala.annotation.elidable

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
// TODO: Rename this trait to Layer and move to a separate library
trait BufferedLayer extends Layer {

  private[deeplearning] val cache =
    java.util.Collections.synchronizedMap(new java.util.IdentityHashMap[BatchId.Aux[Input], BufferedBatch](1))

  protected trait ReferenceCount extends BatchId with Batch { this: BufferedBatch =>

    /**
      * Returns a wrapped [[Batch]] able to detect error of closing more than once if ASSERTION is enabled,
      * or returns this [[ReferenceCount]] itself when ASSERTION is disabled hence no check.
      */
    override final def open(): Open = {
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
        override type Delta = ReferenceCount.this.Delta
        override type Data = ReferenceCount.this.Data

        override def backward(delta: Delta) = ReferenceCount.this.backward(delta)

        override def value = ReferenceCount.this.value

        private var closed = false

        override protected def finalize(): Unit = {
          assert(closed)
        }

        override def close(): Unit = {
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

    private[BufferedLayer] var count: Int = 0

    protected def flush(): Unit

    def input: BatchId.Aux[Input]

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

    override final def backward(delta: Delta): Unit = {
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

    override final def backward(delta: Delta): Unit = {
      synchronized {
        currentDelta = currentDelta |+| Some(delta)
      }
    }
  }

  type Output = BufferedBatch#Open

  protected type BufferedBatch <: ReferenceCount

  /**
    * Performs the underlying forward pass.
    *
    * @return a [[Batch]] that will be cached for subsequent [[#forward]]
    */
  protected def rawForward(input: BatchId.Aux[Input]): BufferedBatch

  override final def forward(input: BatchId.Aux[Input]): BatchId.Aux[Output] = {
    cache.get(input) match {
      case null =>
        rawForward(input)
      case sharedBatch =>
        sharedBatch
    }
  }
}
