package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.Batch._
import cats.{~> => _, _}

import scala.language.existentials
import scala.language.implicitConversions
import scala.language.higherKinds
import cats.implicits._
import shapeless.DepFn1

import scala.annotation.elidable
import scalaz.Leibniz.===
import scalaz.{Liskov, ~>}
import scalaz.Liskov.<~<

private[deepLearning] sealed trait LowPriorityAst {
  import Ast._
  implicit def outputPairAst2[NN[_ <: Batch], OutputPair <: Batch, Input <: Batch](
      implicit d: NN[OutputPair] <~< WidenAst[Input, OutputPair#Widen])
    : IsAst[NN[OutputPair], Input, OutputPair#Data, OutputPair#Delta] = {
    d.andThen(outputPairAst)
  }

}

// TODO: Split this file into multiple modules
object Ast extends LowPriorityAst {

  /** @template */
  type WidenAst[-Input0 <: Batch, +Output0 <: Batch] =
    Ast {
      type Input >: Input0
      type Output <: Output0
    }

  type IsAst[Ast, Input <: Batch, OutputData, OutputDelta] =
    Ast <~< WidenAst[Input, WidenBatch[OutputData, OutputDelta]]

  implicit def outputPairAst[Input <: Batch, OutputPair <: Batch]
    : IsAst[WidenAst[Input, OutputPair#Widen], Input, OutputPair#Data, OutputPair#Delta] = {

    val batchProve = Batch.proveWiden[OutputPair]

    type NN[+Output] = WidenAst[Input, Output with Batch]

    Liskov.co[NN, OutputPair#Widen, WidenBatch[OutputPair#Data, OutputPair#Delta]](batchProve)

  }

  trait Cached extends Ast {

    private val cache =
      java.util.Collections.synchronizedMap(new java.util.IdentityHashMap[Input, SharedBatch](1))

    protected trait ReferenceCount extends Batch { this: SharedBatch =>

      /**
        * Returns a wrapped [[Batch]] able to detect error of closing more than once if ASSERTION is enabled,
        * or returns this [[ReferenceCount]] itself when ASSERTION is disabled hence no check.
        */
      private[Cached] def maybeCheckIfCloseOnlyOnce: Widen = {

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

        Option(checkIfCloseOnlyOnce).getOrElse(widen)
      }

      private[Cached] var count: Int = 1

      protected def flush(): Unit

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

    type Output = SharedBatch#Widen

    protected type SharedBatch <: ReferenceCount

    /**
      * Performs the underlying forward pass.
      *
      * @return a [[Batch]] that will be cached for subsequent [[#forward]]
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
      sharedBatch.maybeCheckIfCloseOnlyOnce
    }
  }

}

trait Ast {

  import Ast._

  type Input <: Batch

  type Output <: Batch

  def forward(input: Input): Output

}
