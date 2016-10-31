package com.thoughtworks.deepLearning.any.ast

import cats._
import com.thoughtworks.deepLearning.{Batch, Differentiable}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Throw(throwable: Eval[Throwable]) extends Differentiable with Batch {
  type Input = Batch
  type Output = this.type
  type Data = scala.Nothing
  type Delta = scala.Any

  override def forward(input: Input): Output = this

  override def backward(delta: Delta): Unit = {}

  override def value: Data = {
    throw throwable.value
  }

  override def close(): Unit = {}
}
