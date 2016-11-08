package com.thoughtworks.deepLearning.any.ast

import com.thoughtworks.deepLearning.core.DifferentiableFunction._
import com.thoughtworks.deepLearning.core.Differentiable._
import com.thoughtworks.deepLearning.core.DifferentiableFunction
import com.thoughtworks.deepLearning.core.{Differentiable, DifferentiableFunction}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Identity[Input0 <: Differentiable]() extends DifferentiableFunction {
  type Input = Input0
  type Output = Input0

  override def forward(input: Input): Output = {
    input
  }
}
