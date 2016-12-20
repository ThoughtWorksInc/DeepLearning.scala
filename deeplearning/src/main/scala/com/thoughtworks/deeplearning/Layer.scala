package com.thoughtworks.deeplearning

import language.existentials
import language.implicitConversions
import language.higherKinds

object Layer {

  /** @template */
  type Aux[-Input0 <: Batch, +Output0 <: Batch] =
    Layer {
      type Input >: Input0
      type Output <: Output0
    }

}

trait Layer {

  type Input <: Batch

  type Output <: Batch

  def forward(input: Input): Output

}
