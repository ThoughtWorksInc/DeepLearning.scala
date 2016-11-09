package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.Differentiable._
import cats._

import scala.language.existentials
import scala.language.implicitConversions
import scala.language.higherKinds
import cats.implicits._
import shapeless.{DepFn1, Lazy}
import any.ast._
import com.thoughtworks.deepLearning.DifferentiableFunction.ToAst

import scala.annotation.elidable
sealed trait LowLowLowPriortyDifferentiableFunction {

  implicit def subtypingToAst[Input <: Differentiable, OutputData, OutputDelta, From](
      implicit view: Lazy[From <:< DifferentiableFunction.Ast[Input, Batch[OutputData, OutputDelta]]])
    : ToAst[From, Input, OutputData, OutputDelta] = {
    new ToAst[From, Input, OutputData, OutputDelta] {
      override def apply(value: From) = view.value(value)
    }

  }
}
sealed trait LowLowPriortyDifferentiableFunction extends LowLowLowPriortyDifferentiableFunction {
  this: LowPriortyDifferentiableFunction =>

  import DifferentiableFunction._

  implicit def toAstNN[Input <: Differentiable, NN[_ <: Differentiable], OutputPair <: Differentiable](
      implicit nn: Lazy[NN[OutputPair] <:< DifferentiableFunction.Ast[Input, OutputPair#Batch]])
    : ToAst[NN[OutputPair], Input, OutputPair#Data, OutputPair#Delta] = {
    new ToAst[NN[OutputPair], Input, OutputPair#Data, OutputPair#Delta] {
      override def apply(value: NN[OutputPair]): Ast[Input, Batch[OutputPair#Data, OutputPair#Delta]] = {
        toAstPair.apply(nn.value(value))
      }
    }
  }

}

sealed trait LowPriortyDifferentiableFunction extends LowLowPriortyDifferentiableFunction {

  import DifferentiableFunction._

  implicit def toAstPair[Input <: Differentiable, OutputPair <: Differentiable]
    : ToAst[DifferentiableFunction.Ast[Input, OutputPair#Batch], Input, OutputPair#Data, OutputPair#Delta] = {
    // I can't prove this because the lack of for-all type in Scala language. Force it as a workaround.
    new ToAst[DifferentiableFunction.Ast[Input, OutputPair#Batch], Input, OutputPair#Data, OutputPair#Delta] {
      override def apply(value: Ast[Input, OutputPair#Batch]): Ast[Input, Batch[OutputPair#Data, OutputPair#Delta]] =
        value.asInstanceOf[Ast[Input, Batch[OutputPair#Data, OutputPair#Delta]]]
    }
  }

}

object DifferentiableFunction extends LowPriortyDifferentiableFunction {

  /** @template */
  type Ast[-Input0 <: Differentiable, +Output0 <: Differentiable] =
    DifferentiableFunction {
      type Input >: Input0
      type Output <: Output0
    }

  trait ToAst[T, Input <: Differentiable, OutputData, OutputDelta] {
    def apply(value: T): DifferentiableFunction.Ast[Input, Differentiable.Batch[OutputData, OutputDelta]]
  }

  implicit def toAst[Input <: Differentiable, OutputData, OutputDelta]
    : ToAst[DifferentiableFunction.Ast[Input, Differentiable.Batch[OutputData, OutputDelta]],
            Input,
            OutputData,
            OutputDelta] = {
    new ToAst[DifferentiableFunction.Ast[Input, Differentiable.Batch[OutputData, OutputDelta]],
              Input,
              OutputData,
              OutputDelta] {
      override def apply(
          value: Ast[Input, Batch[OutputData, OutputDelta]]): Ast[Input, Batch[OutputData, OutputDelta]] = value
    }

  }

  trait Cached extends DifferentiableFunction {

    private val cache =
      java.util.Collections.synchronizedMap(new java.util.IdentityHashMap[Input, SharedBatch](1))

    protected trait ReferenceCount extends Differentiable { this: SharedBatch =>

      /**
        * Returns a wrapped [[Differentiable]] able to detect error of closing more than once if ASSERTION is enabled,
        * or returns this [[ReferenceCount]] itself when ASSERTION is disabled hence no check.
        */
      private[Cached] def maybeCheckIfCloseOnlyOnce: Batch = {

        // Returns a [[Differentiable]] able to detect error of closing more than once.
        @elidable(elidable.ASSERTION)
        def checkIfCloseOnlyOnce = new Differentiable {
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

        Option(checkIfCloseOnlyOnce).getOrElse(toBatch)
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

    type Output = SharedBatch#Batch

    protected type SharedBatch <: ReferenceCount

    /**
      * Performs the underlying forward pass.
      *
      * @return a [[Differentiable]] that will be cached for subsequent [[#forward]]
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

trait DifferentiableFunction {

  import DifferentiableFunction._

  type Input <: Differentiable

  type Output <: Differentiable

  def forward(input: Input): Output

}
