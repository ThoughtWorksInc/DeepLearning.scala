package com.thoughtworks.deepLearning.any.layers

import com.thoughtworks.deepLearning.{Batch, BatchId, Layer}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Compose[Input0 <: Batch, Temporary <: Batch, Output0 <: Batch](
    leftOperand: Layer.Aux[Temporary, Output0],
    rightOperand: Layer.Aux[Input0, Temporary])
    extends Layer {
  override type Input = Input0
  override type Output = Output0

  override def forward(input: BatchId.Aux[Input]): BatchId.Aux[Output] = {
    leftOperand.forward(rightOperand.forward(input))
  }
}
