package com.thoughtworks.deepLearning
package coproduct.layer

import cats.Eval
import com.thoughtworks.deepLearning.boolean.utilities.BooleanMonoidBatch
import com.thoughtworks.deepLearning.utilities.CloseableOnce

final case class IsInl[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
TailDelta <: shapeless.Coproduct](
    ccons: Layer.Aux[Input0, Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
) extends Layer {

  final class Output private[IsInl] (
      upstream: Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
      extends BooleanMonoidBatch
      with Batch
      with CloseableOnce {

    val value = upstream.value match {
      case shapeless.Inl(_) => Eval.now(true)
      case shapeless.Inr(_) => Eval.now(false)
    }

    override def backward(delta: Eval[scala.Boolean]): Unit = {}

    override def close(): Unit = {
      super.close()
      upstream.close()
    }
  }

  type Input = Input0

  override def forward(input: BatchId.Aux[Input]) = new BatchId {
    override type Open = Output
    override def open() = new Output(ccons.forward(input).open())
  }
}
