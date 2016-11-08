package com.thoughtworks.deepLearning.double.ast

import cats._
import com.thoughtworks.deepLearning.core.{Differentiable, DifferentiableFunction, LearningRate}
import com.thoughtworks.deepLearning.double.utilities.DoubleMonoidBatch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Weight(var rawValue: scala.Double)(implicit learningRate: LearningRate)
    extends DifferentiableFunction
    with DoubleMonoidBatch {
  override type Input = Differentiable
  override type Output = Differentiable.Batch[Data, Delta]

  override def forward(any: Input) = this

  override def backward(delta: Delta): Unit = {
    rawValue -= delta.value * learningRate()
  }

  override def value = Eval.now(rawValue)

  override def close(): Unit = {}

}
