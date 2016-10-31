package com.thoughtworks.deepLearning.any.ast

import com.thoughtworks.deepLearning.{Batch, Differentiable}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Identity[Data, Delta]() extends Differentiable {
  type Input = Batch.Aux[Data, Delta]
  type Output = Batch.Aux[Data, Delta]

  override def forward(input: Input): Output = {
    input
  }
}
