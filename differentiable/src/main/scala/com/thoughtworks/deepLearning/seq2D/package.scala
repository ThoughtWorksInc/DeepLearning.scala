package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.Ast.WidenAst
import com.thoughtworks.deepLearning.seq2D.ast.Get

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object seq2D {

  /** @template */
  type Seq2D = utilities.Seq2D

  implicit final class Seq2DOps[Input <: Batch](differentiable: WidenAst[Input, Seq2D#Widen]) {
    def apply(rowIndex: Int, columnIndex: Int) = Get(differentiable, rowIndex, columnIndex)
  }
}
