package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.Differentiable.Batch
import com.thoughtworks.deepLearning.DifferentiableFunction.Ast
import com.thoughtworks.deepLearning.seq2D.ast.Get
import com.thoughtworks.deepLearning.double.utilities.Double
import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object seq2D {

  /** @template */
  type Seq2D = utilities.Seq2D

  final class Seq2DOps[Input <: Differentiable](differentiable: DifferentiableFunction.Ast[Input, Seq2D#Batch]) {
    def apply(rowIndex: Int, columnIndex: Int): Ast[Input, Double#Batch] =
      Get(differentiable, rowIndex, columnIndex)
  }

  implicit def toSeq2DOps[A, Input <: Differentiable, OutputData, OutputDelta](a: A)(
      implicit toAst: ToAst.OfType[A, Input, Seq2D]): Seq2DOps[Input] = {
    new Seq2DOps(toAst(a))
  }

}
