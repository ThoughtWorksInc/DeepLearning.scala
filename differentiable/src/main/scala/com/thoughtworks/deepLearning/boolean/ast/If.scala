package com.thoughtworks.deepLearning.boolean.ast

import com.thoughtworks.deepLearning.{Batch, Ast}
import com.thoughtworks.deepLearning.Ast._
import com.thoughtworks.deepLearning.boolean.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class If[Input0 <: Batch, Output0 <: Batch](condition: WidenAst[Input0, Boolean#Widen],
                                                       `then`: WidenAst[Input0, Output0],
                                                       `else`: WidenAst[Input0, Output0])
    extends Ast {
  override type Input = Input0
  override type Output = Output0

  override def forward(input: Input0): Output0 = {
    val conditionForwardPass = condition.forward(input)
    if (conditionForwardPass.value.value) {
      `then`.forward(input)
    } else {
      `else`.forward(input)
    }
  }
}
