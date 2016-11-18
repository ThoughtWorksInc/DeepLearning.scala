package com.thoughtworks.deepLearning
package coproduct.ast

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Inr[Input0 <: Batch, TailData <: shapeless.Coproduct, TailDelta <: shapeless.Coproduct](
    tail: NeuralNetwork.Aux[Input0, Batch.Aux[TailData, TailDelta]])
    extends NeuralNetwork {

  type Input = Input0

  final class Output private[Inr] (tailBatch: Batch.Aux[TailData, TailDelta]) extends Batch.Unshared {
    def value = shapeless.Inr(tailBatch.value: TailData)

    type Data = shapeless.Inr[Nothing, TailData]
    type Delta = shapeless.:+:[scala.Any, TailDelta]

    override def backward(delta: shapeless.:+:[scala.Any, TailDelta]): Unit = {
      delta match {
        case shapeless.Inr(tailDelta) => tailBatch.backward(tailDelta)
        case shapeless.Inl(_) =>
      }
    }

    override protected def closeUpstreams(): Unit = {
      tailBatch.close()
    }
  }

  override def forward(input: BatchId.Aux[Input]) = new BatchId {
    override type Open = Output
    override def open() = new Output(tail.forward(input).open())
  }

}
