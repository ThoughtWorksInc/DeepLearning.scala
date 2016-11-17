package com.thoughtworks.deepLearning
package coproduct.ast

import com.thoughtworks.deepLearning.NeuralNetwork.Aux
import com.thoughtworks.deepLearning.Batch.Aux
import com.thoughtworks.deepLearning.{NeuralNetwork, Batch}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Head[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
TailDelta <: shapeless.Coproduct](
    ccons: NeuralNetwork.Aux[Input0, Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
) extends NeuralNetwork {

  final class Output private[Head] (
      upstream: Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
      extends Batch {
    override type Data = HeadData
    override type Delta = HeadDelta
    type Input >: Input0

    val value =
      upstream.value.asInstanceOf[shapeless.Inl[HeadData, TailData]].head

    override def backward(delta: Delta): Unit = {
      upstream.backward(shapeless.Inl(delta))
    }

    override def close(): Unit = {
      upstream.close()
    }

  }

  type Input = Input0

  override def forward(input: BatchId.Aux[Input]) = new BatchId {
    override type Open = Output
    override def open() = new Output(ccons.forward(input).open())
  }

}
