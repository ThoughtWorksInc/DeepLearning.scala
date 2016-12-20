package com.thoughtworks.deeplearning.dsl.layers

import com.thoughtworks.deeplearning.{Batch, Layer}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Compose[Input0 <: Batch, Temporary <: Batch, Output0 <: Batch](
    leftOperand: Layer.Aux[Temporary, Output0],
    rightOperand: Layer.Aux[Input0, Temporary])
    extends Layer {
  override type Input = Input0
  override type Output = Output0

  override def forward(input: Input): Output = {
    val tmpBatch = rightOperand.forward(input)
    try {
      leftOperand.forward(tmpBatch)
    } finally {
      tmpBatch.close()
    }
  }
}
