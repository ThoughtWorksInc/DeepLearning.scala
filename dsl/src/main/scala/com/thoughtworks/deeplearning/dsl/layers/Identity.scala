package com.thoughtworks.deeplearning
package dsl.layers

import com.thoughtworks.deeplearning.Layer.Batch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Identity[Data, Delta]() extends Layer {
  type Input = Batch.Aux[Data, Delta]
  type Output = Batch.Aux[Data, Delta]

  override def forward(input: Input): Output = {
    input.addReference()
  }
}
