package com.thoughtworks.deeplearning
package hlist.layers

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Tail[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList,
TailDelta <: shapeless.Coproduct](
    operand: Layer.Aux[Input0, Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
) extends Layer {
  override type Input = Input0

  final class Output private[Tail] (
      upstream: Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
      extends Batch
      with com.thoughtworks.deeplearning.utilities.CloseableOnce {
    override def backward(delta: Delta): Unit = {
      upstream.backward(shapeless.Inr(delta))
    }

    override def value: Data = {
      upstream.value.tail
    }

    override def close(): Unit = {
      super.close()
      upstream.close()
    }

    override type Data = TailData
    override type Delta = TailDelta

    override def addReference() = new Output(upstream.addReference())
  }
  override def forward(input: Input) = new Output(operand.forward(input))

}
