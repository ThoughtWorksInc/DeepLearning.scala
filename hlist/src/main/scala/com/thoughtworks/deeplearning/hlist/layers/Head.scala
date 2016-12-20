package com.thoughtworks.deeplearning
package hlist.layers

import shapeless._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Head[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList,
TailDelta <: shapeless.Coproduct](
    operand: Layer.Aux[Input0, Batch.Aux[::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
) extends Layer {
  override type Input = Input0

  final class Output private[Head] (
      upstream: Batch.Aux[::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
      extends Batch
      with com.thoughtworks.deeplearning.utilities.CloseableOnce {
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

    override def addReference() = new Output(upstream.addReference())
  }
  override def forward(input: Input) = new Output(operand.forward(input))

}
