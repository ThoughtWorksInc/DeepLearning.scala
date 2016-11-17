package com.thoughtworks.deepLearning.any.ast

import com.thoughtworks.deepLearning.{Batch, BatchId, NeuralNetwork}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Literal[Data0](value0: Data0) extends NeuralNetwork with Batch with BatchId {
  override type Data = Data0
  override type Delta = scala.Any
  override type Input = Batch
  override type Output = Batch.Aux[Data, Delta]
  override type Open = Output

  override def value: Data = value0

  override def forward(input: BatchId.Aux[Input]) = {
    this
  }

  override def backward(delta: Delta): Unit = {}

  override def close(): Unit = {}

  override def open() = this
}
