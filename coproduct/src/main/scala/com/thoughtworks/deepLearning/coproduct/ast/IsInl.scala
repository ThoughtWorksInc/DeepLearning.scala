package com.thoughtworks.deepLearning
package coproduct.ast

import cats.Eval
import com.thoughtworks.deepLearning.NeuralNetwork.Aux
import com.thoughtworks.deepLearning.Batch.Aux
import com.thoughtworks.deepLearning.boolean.utilities.BooleanMonoidBatch
import com.thoughtworks.deepLearning.{NeuralNetwork, Batch}

final case class IsInl[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
TailDelta <: shapeless.Coproduct](
    ccons: NeuralNetwork.Aux[Input0, Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
) extends NeuralNetwork {

  final class Output private[IsInl] (
      upstream: Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
      extends BooleanMonoidBatch with Batch.Unshared {


    val value = upstream.value match {
      case shapeless.Inl(_) => Eval.now(true)
      case shapeless.Inr(_) => Eval.now(false)
    }

    override def backward(delta: Eval[scala.Boolean]): Unit = {}

    override protected def closeUpstreams(): Unit = {
      upstream.close()
    }
  }

  type Input = Input0

  override def forward(input: BatchId.Aux[Input]) = new BatchId {
    override type Open = Output
    override def open() = new Output(ccons.forward(input).open())
  }
}
