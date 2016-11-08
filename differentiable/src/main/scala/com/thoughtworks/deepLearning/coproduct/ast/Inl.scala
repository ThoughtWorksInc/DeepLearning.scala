package com.thoughtworks.deepLearning
package coproduct.ast

import com.thoughtworks.deepLearning.core.{Differentiable, DifferentiableFunction}
import com.thoughtworks.deepLearning.core.DifferentiableFunction.Ast
import com.thoughtworks.deepLearning.core.Differentiable.Batch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Inl[Input0 <: Differentiable, HeadData, HeadDelta](head: DifferentiableFunction.Ast[Input0, Differentiable.Batch[HeadData, HeadDelta]])
    extends DifferentiableFunction {

  type Input = Input0

  final class Output private[Inl] (headBatch: Differentiable.Batch[HeadData, HeadDelta]) extends Differentiable {
    def value = shapeless.Inl(headBatch.value: HeadData)

    type Data = shapeless.Inl[HeadData, Nothing]
    type Delta = shapeless.:+:[HeadDelta, shapeless.Coproduct]

    override def backward(delta: shapeless.:+:[HeadDelta, shapeless.Coproduct]): Unit = {
      delta match {
        case shapeless.Inl(headDelta) => headBatch.backward(headDelta)
        case shapeless.Inr(_) =>
      }
    }

    override def close(): Unit = {
      headBatch.close()
    }
  }

  override def forward(input: Input0): Output = {
    new Output(head.forward(input))
  }

}
