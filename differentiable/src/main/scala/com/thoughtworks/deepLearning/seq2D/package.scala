package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.core.{Differentiable, DifferentiableFunction}
import com.thoughtworks.deepLearning.core.DifferentiableFunction.Ast
import com.thoughtworks.deepLearning.seq2D.ast.Get
import com.thoughtworks.deepLearning.double.utilities.Double

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object seq2D {

  /** @template */
  type Seq2D = utilities.Seq2D

  implicit final class Seq2DOps[Input <: Differentiable](differentiable: DifferentiableFunction.Ast[Input, Seq2D#Batch]) {
    def apply(rowIndex: Int, columnIndex: Int): DifferentiableFunction.Ast[Input, Double#Batch] =
      Get(differentiable, rowIndex, columnIndex)
  }
}
