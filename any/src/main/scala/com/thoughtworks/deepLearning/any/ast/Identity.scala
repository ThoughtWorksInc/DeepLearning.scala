package com.thoughtworks.deepLearning.any.ast

import com.thoughtworks.deepLearning.{Batch, BatchId, NeuralNetwork}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Identity[Input0 <: Batch]() extends NeuralNetwork {
  type Input = Input0
  type Output = Input0

  override def forward(input: BatchId.Aux[Input]): BatchId.Aux[Output] = {
    input
  }
}
