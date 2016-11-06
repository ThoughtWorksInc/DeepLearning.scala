package com.thoughtworks.deepLearning.seq2D.ast

import cats._
import com.thoughtworks.deepLearning.Ast._
import com.thoughtworks.deepLearning.{Ast, Batch}
import com.thoughtworks.deepLearning.array2D.utilities._
import com.thoughtworks.deepLearning.double.utilities.DoubleMonoidBatch
import com.thoughtworks.deepLearning.seq2D.utilities.Seq2D

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Get[Input0 <: Batch](operand0: WidenAst[Input0, Seq2D#Widen], i: Int, j: Int) extends Ast {
  final class Output private[Get] (upstream: Seq2D#Widen) extends DoubleMonoidBatch {

    override def backward(delta: Eval[scala.Double]): Unit = {
      upstream.backward(delta.map((i, j, _)))
    }

    override def close(): Unit = {
      upstream.close()
    }

    override val value: Eval[scala.Double] = {
      upstream.value.map { v =>
        v(i)(j)
      }.memoize
    }

  }
  override type Input = Input0

  // TODO: Support tail Int
  override def forward(input: Input): Output = {
    new Output(operand0.forward(input))
  }

}
