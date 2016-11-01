package com.thoughtworks.deepLearning
package coproduct.ast

import com.thoughtworks.deepLearning.Ast.WidenAst
import com.thoughtworks.deepLearning.Batch.WidenBatch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Inr[Input0 <: Batch, TailData <: shapeless.Coproduct, TailDelta <: shapeless.Coproduct](
    tail: WidenAst[Input0, WidenBatch[TailData, TailDelta]])
    extends Ast {

  type Input = Input0

  final class Output private[Inr] (tailBatch: WidenBatch[TailData, TailDelta]) extends Batch {
    def value = shapeless.Inr(tailBatch.value: TailData)

    type Data = shapeless.Inr[Nothing, TailData]
    type Delta = shapeless.:+:[scala.Any, TailDelta]

    override def backward(delta: shapeless.:+:[scala.Any, TailDelta]): Unit = {
      delta match {
        case shapeless.Inr(tailDelta) => tailBatch.backward(tailDelta)
        case shapeless.Inl(_) =>
      }
    }

    override def close(): Unit = {
      tailBatch.close()
    }
  }

  override def forward(input: Input0): Output = {
    new Output(tail.forward(input))
  }

}
