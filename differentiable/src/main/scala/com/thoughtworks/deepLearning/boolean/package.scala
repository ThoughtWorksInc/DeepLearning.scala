package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.Ast._
import com.thoughtworks.deepLearning.any.Any
import cats.Eval
import com.thoughtworks.deepLearning.array2D.ast.{Dot, Negative}
import com.thoughtworks.deepLearning.boolean.ast.If

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object boolean {

  /** @template */
  type Boolean = utilities.Boolean

  implicit final class BooleanOps[Input <: Batch](differentiable: WidenAst[Input, Boolean#Widen]) {

    def `if`[ThatInput <: Input, Output <: Batch](`then`: WidenAst[ThatInput, Output])(
        `else`: WidenAst[ThatInput, Output]): WidenAst[ThatInput, Output] = {
      If[ThatInput, Output](differentiable, `then`, `else`)
    }

  }

}
