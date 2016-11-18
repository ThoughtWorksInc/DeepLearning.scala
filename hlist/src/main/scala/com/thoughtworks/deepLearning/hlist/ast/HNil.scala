package com.thoughtworks.deepLearning.hlist.ast

import com.thoughtworks.deepLearning.{Batch, BatchId, NeuralNetwork}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
case object HNil extends NeuralNetwork with Batch with BatchId {
  override type Input = Batch

  override type Data = shapeless.HNil

  override type Delta = shapeless.CNil

  override type Output = Batch.Aux[Data, Delta]
  override type Open = Output

  override def open() = this

  override def forward(input: BatchId.Aux[Input]) = this

  override def backward(delta: Delta): Unit = {}

  override def value = shapeless.HNil

  override protected def close(): Unit = {}
}
