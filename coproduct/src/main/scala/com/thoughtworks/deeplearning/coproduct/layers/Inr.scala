package com.thoughtworks.deeplearning
package coproduct.layers

import com.thoughtworks.deeplearning.Layer.Batch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Inr[Input0 <: Batch, TailData <: shapeless.Coproduct, TailDelta <: shapeless.Coproduct](
    operand: Layer.Aux[Input0, Batch.Aux[TailData, TailDelta]])
    extends Layer {

  type Input = Input0

  final class Output private[Inr] (upstream: Batch.Aux[TailData, TailDelta])
      extends Batch
      with com.thoughtworks.deeplearning.Layer.CloseableOnce {
    def value = shapeless.Inr(upstream.value: TailData)

    type Data = shapeless.Inr[Nothing, TailData]
    type Delta = shapeless.:+:[Any, TailDelta]

    override def backward(delta: shapeless.:+:[Any, TailDelta]): Unit = {
      delta match {
        case shapeless.Inr(tailDelta) => upstream.backward(tailDelta)
        case shapeless.Inl(_) =>
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
