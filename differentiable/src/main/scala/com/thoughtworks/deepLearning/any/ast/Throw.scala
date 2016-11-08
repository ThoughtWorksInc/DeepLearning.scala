package com.thoughtworks.deepLearning.any.ast

import com.thoughtworks.deepLearning.core.DifferentiableFunction._
import com.thoughtworks.deepLearning.core.Differentiable._
import cats._
import com.thoughtworks.deepLearning.core.DifferentiableFunction
import com.thoughtworks.deepLearning.core.{Differentiable, DifferentiableFunction}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Throw(throwable: () => Throwable) extends DifferentiableFunction with Differentiable {
  type Input = Differentiable
  type Output = this.type
  type Data = scala.Nothing
  type Delta = scala.Any

  override def forward(input: Input): Output = this

  override def backward(delta: Delta): Unit = {}

  override def value: Data = {
    throw throwable()
  }

  override def close(): Unit = {}
}
