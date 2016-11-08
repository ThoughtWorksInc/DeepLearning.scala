package com.thoughtworks.deepLearning.hlist.ast

import com.thoughtworks.deepLearning.core.DifferentiableFunction
import com.thoughtworks.deepLearning.core.{Differentiable, DifferentiableFunction}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Tail[Input0 <: Differentiable, HeadData, HeadDelta, TailData <: shapeless.HList,
TailDelta <: shapeless.Coproduct](
    differentiableHCons: DifferentiableFunction.Ast[
      Input0,
      Differentiable.Batch[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
) extends DifferentiableFunction {
  override type Input = Input0

  final class Output private[Tail] (
      upstream: Differentiable.Batch[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
      extends Differentiable {
    override def backward(delta: Delta): Unit = {
      upstream.backward(shapeless.Inr(delta))
    }

    override def value: Data = {
      upstream.value.tail
    }

    override def close(): Unit = {
      upstream.close()
    }

    override type Data = TailData
    override type Delta = TailDelta
  }

  override def forward(input: Input) = {
    new Output(differentiableHCons.forward(input))
  }
}
