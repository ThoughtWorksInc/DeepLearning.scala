package com.thoughtworks.deepLearning.any.ast

import com.thoughtworks.deepLearning.{Batch, Ast}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Compose[Input0 <: Batch, Temporary <: Batch, Output0 <: Batch](
    leftOperand: Ast.Aux[Temporary, Output0],
    rightOperand: Ast.Aux[Input0, Temporary])
    extends Ast {
  override type Input = Input0
  override type Output = Output0

  override def forward(input: Input0): Output0 = {
    leftOperand.forward(rightOperand.forward(input))
  }
}
