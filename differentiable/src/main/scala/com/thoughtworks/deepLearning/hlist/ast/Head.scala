package com.thoughtworks.deepLearning.hlist.ast

import com.thoughtworks.deepLearning.{Differentiable, DifferentiableFunction}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Head[Input0 <: Differentiable, HeadData, HeadDelta, TailData <: shapeless.HList,
TailDelta <: shapeless.Coproduct](
    differentiableHCons: DifferentiableFunction.Ast[
      Input0,
      Differentiable.Batch[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
) extends DifferentiableFunction {
  override type Input = Input0

  final class Output private[Head] (
      upstream: Differentiable.Batch[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
      extends Differentiable {
    override def backward(delta: Delta): Unit = {
      upstream.backward(shapeless.Inl(delta))
    }

    override def value: Data = {
      upstream.value.head
    }

    override type Data = HeadData
    override type Delta = HeadDelta

    override def close(): Unit = {
      upstream.close()
    }

  }

  override def forward(input: Input) = {
    new Output(differentiableHCons.forward(input))
  }
}
