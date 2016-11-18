package com.thoughtworks.deepLearning.hlist.layer

import com.thoughtworks.deepLearning.{Batch, BatchId, Layer}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Head[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList,
TailDelta <: shapeless.Coproduct](
    differentiableHCons: Layer.Aux[Input0,
                                   Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
) extends Layer {
  override type Input = Input0

  final class Output private[Head] (
      upstream: Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
      extends Batch
      with com.thoughtworks.deepLearning.utilities.CloseableOnce {
    override def backward(delta: Delta): Unit = {
      upstream.backward(shapeless.Inl(delta))
    }

    override def value: Data = {
      upstream.value.head
    }

    override type Data = HeadData
    override type Delta = HeadDelta

    override def close(): Unit = {
      super.close()
      upstream.close()
    }

  }

  override def forward(input: BatchId.Aux[Input]) = new BatchId {
    override type Open = Output
    override def open() = new Output(differentiableHCons.forward(input).open())
  }
}
