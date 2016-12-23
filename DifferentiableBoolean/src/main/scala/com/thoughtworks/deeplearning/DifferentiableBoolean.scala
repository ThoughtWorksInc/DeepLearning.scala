package com.thoughtworks.deeplearning

import cats.{Eval, Monoid}
import cats.implicits._
import com.thoughtworks.deeplearning.Layer.{Aux, Batch}
import com.thoughtworks.deeplearning.Lift._
import shapeless.Lub

import language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableBoolean {

  private[deeplearning] trait BooleanMonoidBatch extends Batch {

    override type Data = Boolean

    override type Delta = Boolean

    protected final def monoid = new Monoid[Delta] {
      override def empty = false

      override def combine(x: Delta, y: Delta) = x ^ y
    }

  }

  object Layers {

    final case class If[Input0 <: Batch, OutputData0, OutputDelta0](
        condition: Layer.Aux[Input0, BooleanPlaceholder.Batch],
        `then`: Layer.Aux[Input0, Batch.Aux[OutputData0, OutputDelta0]],
        `else`: Layer.Aux[Input0, Batch.Aux[OutputData0, OutputDelta0]])
        extends Layer {
      override type Input = Input0
      override type Output = Batch.Aux[OutputData0, OutputDelta0]

      override def forward(input: Input0) = {
        resource.managed(condition.forward(input)).acquireAndGet { conditionBatch =>
          (if (conditionBatch.value) `then` else `else`).forward(input)
        }
      }
    }

    final case class Not[Input0 <: Batch](operand: Layer.Aux[Input0, BooleanPlaceholder.Batch])
        extends BufferedLayer.Unary {

      type BufferedBatch = MonoidBatch with BooleanMonoidBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {

          override val input = input0

        } with MonoidBatch with BooleanMonoidBatch with UnaryBatch {

          override val value = !upstream.value

          override protected def rawBackward(delta: Boolean): Unit = {
            upstream.backward(!delta)
          }

        }
      }
    }

    final case class Weight[Input0 <: Batch](var value: Boolean) extends Layer with BooleanMonoidBatch {
      override type Input = Input0
      override type Output = Weight[Input0]

      override def forward(any: Input) = this

      override def backward(delta: Delta): Unit = {
        value ^= delta
      }

      override def close(): Unit = {}

      override def addReference(): Weight[Input0] = this

    }

  }

  import Layers._

  private[deeplearning] type BooleanPlaceholder = Placeholder[Boolean, Boolean]
  private[deeplearning] val BooleanPlaceholder: BooleanPlaceholder = implicitly

  final class BooleanLayerOps[Input <: Batch](boolean: Layer.Aux[Input, BooleanPlaceholder.Batch]) {

    def `if`[Then,
             Else,
             ThenOutputData,
             ThenOutputDelta,
             ElseOutputData,
             ElseOutputDelta,
             NN,
             OutputData,
             OutputDelta](`then`: Then)(`else`: Else)(
        implicit thenToLayer: ToLayer.Aux[Then, Input, ThenOutputData, ThenOutputDelta],
        elseToLayer: ToLayer.Aux[Else, Input, ElseOutputData, ElseOutputDelta],
        lub: Lub[Layer.Aux[Input, Batch.Aux[ThenOutputData, ThenOutputDelta]],
                 Layer.Aux[Input, Batch.Aux[ElseOutputData, ElseOutputDelta]],
                 NN],
        commonToLayer: ToLayer.Aux[NN, Input, OutputData, OutputDelta]
    ): Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = {
      If[Input, OutputData, OutputDelta](boolean,
                                         commonToLayer(lub.left(thenToLayer(`then`))),
                                         commonToLayer(lub.right(elseToLayer(`else`))))
    }

  }

  implicit def toBooleanLayerOps[From, Input <: Batch](from: From)(
      implicit toLayer: ToLayer.OfType[From, Input, BooleanPlaceholder]): BooleanLayerOps[Input] = {
    new BooleanLayerOps[Input](toLayer(from))
  }

  implicit def liftBoolean: Lift.Aux[Boolean, Boolean, Boolean] = Lift.fromData

}
