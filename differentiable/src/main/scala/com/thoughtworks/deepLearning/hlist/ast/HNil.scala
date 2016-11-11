package com.thoughtworks.deepLearning.hlist.ast

import com.thoughtworks.deepLearning.Differentiable.Batch
import com.thoughtworks.deepLearning.{Differentiable, DifferentiableFunction}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
case object HNil extends DifferentiableFunction with Differentiable {
  override type Input = Differentiable

  override type Data = shapeless.HNil

  override type Delta = shapeless.CNil

  override type Output = Batch[Data, Delta]

  override def forward(input: Input): Output = this

  override def backward(delta: Delta): Unit = {}

  override def value = shapeless.HNil

  override def close(): Unit = {}
}
