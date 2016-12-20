package com.thoughtworks.deeplearning
package coproduct.layers

import com.thoughtworks.deeplearning.Layer.Batch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Tail[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
TailDelta <: shapeless.Coproduct](
    operand: Layer.Aux[Input0, Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
) extends Layer {

  final class Output private[Tail] (
      upstream: Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
      extends Batch
      with com.thoughtworks.deeplearning.Layer.CloseableOnce {
    override type Data = TailData
    override type Delta = TailDelta

    val value =
      upstream.value.asInstanceOf[shapeless.Inr[TailData, TailData]].tail

    override def backward(delta: Delta): Unit = {
      upstream.backward(shapeless.Inr(delta))
    }

    override def close(): Unit = {
      super.close()
      upstream.close()
    }

    override def addReference() = new Output(upstream.addReference())
  }

  type Input = Input0

  override def forward(input: Input) = new Output(operand.forward(input))

}
