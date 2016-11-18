package com.thoughtworks.deepLearning.any.layer

import com.thoughtworks.deepLearning.{Batch, BatchId, Layer}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Identity[Input0 <: Batch]() extends Layer {
  type Input = Input0
  type Output = Input0

  override def forward(input: BatchId.Aux[Input]): BatchId.Aux[Output] = {
    input
  }
}
