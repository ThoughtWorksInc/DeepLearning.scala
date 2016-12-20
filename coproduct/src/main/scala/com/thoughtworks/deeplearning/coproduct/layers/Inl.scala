package com.thoughtworks.deeplearning
package coproduct.layers

import com.thoughtworks.deeplearning.Layer.Batch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Inl[Input0 <: Batch, HeadData, HeadDelta](operand: Layer.Aux[Input0, Batch.Aux[HeadData, HeadDelta]])
    extends Layer {

  type Input = Input0

  final class Output private[Inl] (upstream: Batch.Aux[HeadData, HeadDelta])
      extends Batch
      with com.thoughtworks.deeplearning.Layer.CloseableOnce {
    def value = shapeless.Inl(upstream.value: HeadData)

    type Data = shapeless.Inl[HeadData, Nothing]
    type Delta = shapeless.:+:[HeadDelta, shapeless.Coproduct]

    override def backward(delta: shapeless.:+:[HeadDelta, shapeless.Coproduct]): Unit = {
      delta match {
        case shapeless.Inl(headDelta) => upstream.backward(headDelta)
        case shapeless.Inr(_) =>
      }
    }

    override def close(): Unit = {
      super.close()
      upstream.close()
    }

    override def addReference() = new Output(upstream.addReference())

  }

  override def forward(input: Input) = new Output(operand.forward(input))

}
