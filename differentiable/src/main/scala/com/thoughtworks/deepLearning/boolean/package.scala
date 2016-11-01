package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.array2D.ast.{Dot, Negative}
import com.thoughtworks.deepLearning.boolean.ast.If

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object boolean {

  type Boolean = {
    type Delta = Eval[scala.Boolean]
    type Data = Eval[scala.Boolean]
  }

  type BooleanBatch = Batch.Aux[Boolean#Data, Boolean#Delta]

  implicit final class BooleanOps[Input <: Batch](differentiable: Ast.Aux[Input, BooleanBatch]) {

    def `if`[ThatInput <: Input, Output <: Batch](`then`: Ast.Aux[ThatInput, Output])(
        `else`: Ast.Aux[ThatInput, Output]):  Ast.Aux[ThatInput, Output] = {
      If[ThatInput, Output](differentiable, `then`, `else`)
    }

  }

}
