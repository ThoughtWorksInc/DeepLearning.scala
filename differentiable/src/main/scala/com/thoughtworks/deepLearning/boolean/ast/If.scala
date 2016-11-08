package com.thoughtworks.deepLearning.boolean.ast

import com.thoughtworks.deepLearning.{Differentiable, DifferentiableFunction}
import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.boolean.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class If[Input0 <: Differentiable, Output0 <: Differentiable](condition: DifferentiableFunction.Ast[Input0, Boolean#Batch],
                                                                         `then`: DifferentiableFunction.Ast[Input0, Output0],
                                                                         `else`: DifferentiableFunction.Ast[Input0, Output0])
    extends DifferentiableFunction {
  override type Input = Input0
  override type Output = Output0

  override def forward(input: Input0): Output0 = {
    val conditionForwardPass = condition.forward(input)
    if (conditionForwardPass.value.value) {
      `then`.forward(input)
    } else {
      `else`.forward(input)
    }
  }
}
