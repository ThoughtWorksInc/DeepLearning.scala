package com.thoughtworks.deepLearning
package coproduct.ast

import cats.Eval
import com.thoughtworks.deepLearning.core.DifferentiableFunction.Ast
import com.thoughtworks.deepLearning.core.Differentiable.Batch
import com.thoughtworks.deepLearning.boolean.utilities.BooleanMonoidBatch
import com.thoughtworks.deepLearning.core.DifferentiableFunction
import com.thoughtworks.deepLearning.core.{Differentiable, DifferentiableFunction}

final case class IsInl[Input0 <: Differentiable, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
TailDelta <: shapeless.Coproduct](
    ccons: DifferentiableFunction.Ast[Input0, Differentiable.Batch[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
) extends DifferentiableFunction {

  final class Output private[IsInl] (
      upstream: Differentiable.Batch[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
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
