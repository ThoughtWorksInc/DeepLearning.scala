package com.thoughtworks.deepLearning.any.ast

import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.Differentiable._
import com.thoughtworks.deepLearning.{Differentiable, DifferentiableFunction}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Compose[Input0 <: Differentiable, Temporary <: Differentiable, Output0 <: Differentiable](
                                                                                 leftOperand: Ast[Temporary, Output0],
                                                                                 rightOperand: Ast[Input0, Temporary])
    extends DifferentiableFunction {
  override type Input = Input0
  override type Output = Output0

  override def forward(input: Input): Output = {
    leftOperand.forward(rightOperand.forward(input))
  }
}
