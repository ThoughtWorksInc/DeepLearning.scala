package com.thoughtworks.deeplearning
package hlist.layers

import shapeless._

final case class HCons[Input0 <: Batch,
                       HeadData,
                       HeadDelta,
                       TailData <: shapeless.HList,
                       TailDelta <: shapeless.Coproduct](
    head: Layer.Aux[Input0, Batch.Aux[HeadData, HeadDelta]],
    tail: Layer.Aux[Input0, Batch.Aux[TailData, TailDelta]]
) extends Layer {
  override type Input = Input0

  final class Output private[HCons] (headBatch: Batch.Aux[HeadData, HeadDelta],
                                     tailBatch: Batch.Aux[TailData, TailDelta])
      extends Batch
      with com.thoughtworks.deeplearning.utilities.CloseableOnce {
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
      super.close()
      headBatch.close()
      tailBatch.close()
    }

    override type Data = HeadData :: TailData
    override type Delta = HeadDelta :+: TailDelta

    override def addReference() = new Output(headBatch.addReference(), tailBatch.addReference())
  }

  override def forward(input: Input) = new Output(head.forward(input), tail.forward(input))

}
