package com.thoughtworks.deepLearning.any.ast

import com.thoughtworks.deepLearning.{Batch, Differentiable}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Compose[A <: Batch, B <: Batch, C <: Batch](leftOperand: Differentiable.Aux[B, C],
                                                             rightOperand: Differentiable.Aux[A, B])
    extends Differentiable {
  override type Input = A
  override type Output = C

  override def forward(input: A): C = {
    leftOperand.forward(rightOperand.forward(input))
  }
}
