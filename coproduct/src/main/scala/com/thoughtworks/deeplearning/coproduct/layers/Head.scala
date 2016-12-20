package com.thoughtworks.deeplearning
package coproduct.layers

import com.thoughtworks.deeplearning.Layer.Batch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Head[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
TailDelta <: shapeless.Coproduct](
    operand: Layer.Aux[Input0, Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
) extends Layer {

  final class Output private[Head] (
      upstream: Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
      extends Batch
      with com.thoughtworks.deeplearning.Layer.CloseableOnce {
    override type Data = HeadData
    override type Delta = HeadDelta

    val value =
      upstream.value.asInstanceOf[shapeless.Inl[HeadData, TailData]].head

    override def backward(delta: Delta): Unit = {
      upstream.backward(shapeless.Inl(delta))
    }

    override def close(): Unit = {
      super.close()
      upstream.close()
    }

    override def addReference() = {
      new Output(upstream.addReference())
    }

  }

  type Input = Input0

  override def forward(input: Input) = new Output(operand.forward(input))

}
