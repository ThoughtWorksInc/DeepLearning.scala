package com.thoughtworks.deepLearning.hlist.ast

import com.thoughtworks.deepLearning.Ast._
import com.thoughtworks.deepLearning.Batch._
import com.thoughtworks.deepLearning.{Batch, Ast}

final case class HCons[Input0 <: Batch,
                       HeadData,
                       HeadDelta,
                       TailData <: shapeless.HList,
                       TailDelta <: shapeless.Coproduct](
                                                          head: WidenAst[Input0, WidenBatch[HeadData, HeadDelta]],
                                                          tail: WidenAst[Input0, WidenBatch[TailData, TailDelta]]
) extends Ast {
  override type Input = Input0

  final class Output private[HCons] (headBatch: WidenBatch[HeadData, HeadDelta],
                                     tailBatch: WidenBatch[TailData, TailDelta])
      extends Batch {
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
