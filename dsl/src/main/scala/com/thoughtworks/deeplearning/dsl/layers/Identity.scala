package com.thoughtworks.deeplearning.dsl.layers

import com.thoughtworks.deeplearning.{Batch, Layer}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Identity[Input0 <: Batch]() extends Layer {
  type Input = Input0
  type Output = Input0

  override def forward(input: Input): Output = {
    input.addReference().asInstanceOf[Output]
  }
}
