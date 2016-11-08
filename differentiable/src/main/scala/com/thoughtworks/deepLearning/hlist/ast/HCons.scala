package com.thoughtworks.deepLearning.hlist.ast

import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.Differentiable._
import com.thoughtworks.deepLearning.{Differentiable, DifferentiableFunction}

final case class HCons[Input0 <: Differentiable,
                       HeadData,
                       HeadDelta,
                       TailData <: shapeless.HList,
                       TailDelta <: shapeless.Coproduct](
                                                          head: Ast[Input0, Batch[HeadData, HeadDelta]],
                                                          tail: Ast[Input0, Batch[TailData, TailDelta]]
) extends DifferentiableFunction {
  override type Input = Input0

  final class Output private[HCons] (headBatch: Batch[HeadData, HeadDelta],
                                     tailBatch: Batch[TailData, TailDelta])
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
