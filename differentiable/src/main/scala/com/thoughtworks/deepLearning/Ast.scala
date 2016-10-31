package com.thoughtworks.deepLearning

import cats._

import scala.language.existentials
import scala.language.implicitConversions
import scala.language.higherKinds
import cats.implicits._

import scala.annotation.elidable

// TODO: Split this file into multiple modules
object Ast {

  type Aux[-Input0 <: Batch, +Output0 <: Batch] =
    Ast {
      type Input >: Input0
      type Output <: Output0
    }

  type FromTypePair[InputTypePair <: {
                      type Data
                      type Delta
                    },
                    OutputTypePair <: {
                      type Data
                      type Delta
                    }] = Ast.Aux[Batch.FromTypePair[InputTypePair], Batch.FromTypePair[OutputTypePair]]

  trait Cached extends Ast {

    private val cache =
      java.util.Collections.synchronizedMap(new java.util.IdentityHashMap[Input, SharedBatch](1))

    private[Cached] sealed trait ReferenceCount extends Batch { this: SharedBatch =>

      private[Cached] type Self = Batch.Aux[Data, Delta]

      /**
        * Returns a [[Batch]] able to detect error of closing more than once.
        */
      @elidable(elidable.ASSERTION)
      private[Cached] def checkIfCloseOnlyOnce: Self = new Batch {
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

      private[Cached] var count: Int = 1

      private[Cached] def flush(): Unit

      /**
        * Closes upstream batch of this [[SharedBatch]]
        */
      protected def closeUpstreams(): Unit

      def input: Input

      override final def close(): Unit = {
        val newCount = synchronized {
          count -= 1
          count
        }
        if (newCount == 0) {
          cache.remove(input)
          flush()
          closeUpstreams()
        }
      }

    }

    protected trait MonoidBatch extends ReferenceCount { this: SharedBatch =>

      private var currentDelta: Delta = monoid.empty

      /**
        * Performs the underlying backward pass with all `delta`s that previously received from [[#backward]].
        */
      protected def rawBackward(delta: Delta): Unit

      implicit protected def monoid: Monoid[Delta]

      override private[Cached] final def flush(): Unit = {
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

      override private[Cached] final def flush(): Unit = {
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

    type Output = SharedBatch#Self

    protected type SharedBatch <: ReferenceCount

    /**
      * Performs the underlying forward pass.
      *
      * @return a [[Batch]] that will cached for subsequent [[#forward]]
      */
    protected def rawForward(input: Input): SharedBatch

    override final def forward(input: Input): Output = {
      val sharedBatch: SharedBatch = cache.get(input) match {
        case null =>
          val sharedBatch = rawForward(input)
          cache.put(input, sharedBatch)
          sharedBatch
        case sharedBatch =>
          sharedBatch.synchronized {
            sharedBatch.count += 1
          }
          sharedBatch
      }

      // When ASSERTION is disabled, fallback to the sharedBatch itself hence no check
      val checked: sharedBatch.Self = Option(sharedBatch.checkIfCloseOnlyOnce).getOrElse(sharedBatch)

      // Workaround for Scala compiler's stupid bug: https://issues.scala-lang.org/browse/SI-10008
      checked.asInstanceOf[Output with Nothing]
    }
  }

}

trait Ast {

  import Ast._

  type Input <: Batch

  type Output <: Batch

  def forward(input: Input): Output

}
