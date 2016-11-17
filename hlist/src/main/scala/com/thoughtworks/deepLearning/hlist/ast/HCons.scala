package com.thoughtworks.deepLearning.hlist.ast

import com.thoughtworks.deepLearning.{Batch, BatchId, NeuralNetwork}

final case class HCons[Input0 <: Batch,
                       HeadData,
                       HeadDelta,
                       TailData <: shapeless.HList,
                       TailDelta <: shapeless.Coproduct](
    head: NeuralNetwork.Aux[Input0, Batch.Aux[HeadData, HeadDelta]],
    tail: NeuralNetwork.Aux[Input0, Batch.Aux[TailData, TailDelta]]
) extends NeuralNetwork {
  override type Input = Input0

  final class Output private[HCons] (headBatch: Batch.Aux[HeadData, HeadDelta],
                                     tailBatch: Batch.Aux[TailData, TailDelta])
      extends Batch.Unshared {
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

    override protected def closeUpstreams(): Unit = {
      headBatch.close()
      tailBatch.close()
    }

    override type Data = shapeless.::[HeadData, TailData]
    override type Delta = shapeless.:+:[HeadDelta, TailDelta]
  }

  override def forward(input: BatchId.Aux[Input]) = new BatchId {
    override type Open = Output
    override def open() = new Output(head.forward(input).open(), tail.forward(input).open())
  }

}
