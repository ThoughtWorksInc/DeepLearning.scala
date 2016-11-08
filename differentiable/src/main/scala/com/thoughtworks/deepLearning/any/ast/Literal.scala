package com.thoughtworks.deepLearning.any.ast

import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.Differentiable._
import com.thoughtworks.deepLearning.{Differentiable, DifferentiableFunction}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Literal[Data0](value0: Data0) extends DifferentiableFunction with Differentiable {
  override type Data = Data0
  override type Delta = scala.Any
  override type Input = Differentiable
  override type Output = Differentiable.Batch[Data, Delta]

  override def value: Data = value0

  override def forward(input: Input): Output = this

  override def backward(delta: Delta): Unit = {}

  override def close(): Unit = {}
}
