package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.DifferentiableFunction._
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

  implicit final class BooleanOps[Input <: Differentiable](differentiable: Ast[Input, Boolean#Widen]) {

    def `if`[ThatInput <: Input, Output <: Differentiable](`then`: Ast[ThatInput, Output])(
        `else`: Ast[ThatInput, Output]): Ast[ThatInput, Output] = {
      If[ThatInput, Output](differentiable, `then`, `else`)
    }

  }

}
