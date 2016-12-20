package com.thoughtworks.deeplearning
package coproduct.layers

import cats.Eval
import com.thoughtworks.deeplearning.BpBoolean.BooleanMonoidBatch
import com.thoughtworks.deeplearning.Layer.{Batch, CloseableOnce}

final case class IsInl[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
TailDelta <: shapeless.Coproduct](
    operand: Layer.Aux[Input0, Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
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

    override def backward(delta: Eval[Boolean]): Unit = {}

    override def close(): Unit = {
      super.close()
      upstream.close()
    }

    override def addReference() = new Output(upstream.addReference())

  }

  type Input = Input0

  override def forward(input: Input) = new Output(operand.forward(input))
}
