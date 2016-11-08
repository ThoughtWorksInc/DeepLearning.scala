package com.thoughtworks.deepLearning.hlist.ast

import com.thoughtworks.deepLearning.core.DifferentiableFunction
import com.thoughtworks.deepLearning.core.{Differentiable, DifferentiableFunction}

final case class HCons[Input0 <: Differentiable,
                       HeadData,
                       HeadDelta,
                       TailData <: shapeless.HList,
                       TailDelta <: shapeless.Coproduct](
    head: DifferentiableFunction.Ast[Input0, Differentiable.Batch[HeadData, HeadDelta]],
    tail: DifferentiableFunction.Ast[Input0, Differentiable.Batch[TailData, TailDelta]]
) extends DifferentiableFunction {
  override type Input = Input0

  final class Output private[HCons] (headBatch: Differentiable.Batch[HeadData, HeadDelta],
                                     tailBatch: Differentiable.Batch[TailData, TailDelta])
      extends Differentiable {
    override def backward(delta: Delta): Unit = {
      delta match {
        case shapeless.Inl(headDelta) =>
          headBatch.backward(headDelta)
        case shapeless.Inr(tailDelta) =>
          tailBatch.backward(tailDelta)
      }
    }

    override def value: Data = {
      headBatch.value :: tailBatch.value
    }

    override def close(): Unit = {
      headBatch.close()
      tailBatch.close()
    }

    override type Data = shapeless.::[HeadData, TailData]
    override type Delta = shapeless.:+:[HeadDelta, TailDelta]
  }

  override def forward(input: Input) = {
    new Output(head.forward(input), tail.forward(input))
  }

}
