package com.thoughtworks.deepLearning
package coproduct.ast

import cats.Eval
import com.thoughtworks.deepLearning.boolean.utilities.BooleanMonoidBatch
import com.thoughtworks.deepLearning.{Batch, Ast}





final case class IsInl[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
TailDelta <: shapeless.Coproduct](
    ccons: Ast.Aux[Input0,
                              Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
) extends Ast {

  final class Output private[IsInl] (
      upstream: Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
      extends BooleanMonoidBatch {

    type Input >: Input0
    val value = upstream.value match {
      case shapeless.Inl(_) => Eval.now(true)
      case shapeless.Inr(_) => Eval.now(false)
    }

    override def backward(delta: Eval[scala.Boolean]): Unit = {}

    override def close(): Unit = {
      upstream.close()
    }
  }

  type Input = Input0

  override def forward(input: Input): Output = {
    new Output(ccons.forward(input))
  }
}
