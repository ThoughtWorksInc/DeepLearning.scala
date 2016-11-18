package com.thoughtworks.deepLearning.hlist.layer

import com.thoughtworks.deepLearning.{Batch, BatchId, Layer}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Tail[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList,
TailDelta <: shapeless.Coproduct](
    differentiableHCons: Layer.Aux[Input0,
                                   Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
) extends Layer {
  override type Input = Input0

  final class Output private[Tail] (
      upstream: Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
      extends Batch.Unshared {
    override def backward(delta: Delta): Unit = {
      upstream.backward(shapeless.Inr(delta))
    }

    override def value: Data = {
      upstream.value.tail
    }

    override protected def closeUpstreams(): Unit = {
      upstream.close()
    }

    override type Data = TailData
    override type Delta = TailDelta
  }

  override def forward(input: BatchId.Aux[Input]) = new BatchId {
    override type Open = Output
    override def open() = new Output(differentiableHCons.forward(input).open())
  }
}
