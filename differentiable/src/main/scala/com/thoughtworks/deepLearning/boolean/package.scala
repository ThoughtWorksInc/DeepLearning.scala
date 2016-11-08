package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.core.DifferentiableFunction._
import com.thoughtworks.deepLearning.any.Any
import cats.Eval
import com.thoughtworks.deepLearning.array2D.ast.{Dot, Negative}
import com.thoughtworks.deepLearning.boolean.ast.If
import com.thoughtworks.deepLearning.core.{Differentiable, DifferentiableFunction}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object boolean {

  /** @template */
  type Boolean = utilities.Boolean

  implicit final class BooleanOps[Input <: Differentiable](differentiable: DifferentiableFunction.Ast[Input, Boolean#Batch]) {

    def `if`[ThatInput <: Input, Output <: Differentiable](`then`: DifferentiableFunction.Ast[ThatInput, Output])(
        `else`: DifferentiableFunction.Ast[ThatInput, Output]): DifferentiableFunction.Ast[ThatInput, Output] = {
      If[ThatInput, Output](differentiable, `then`, `else`)
    }

  }

}
