package com.thoughtworks.deepLearning

import scala.language.implicitConversions
import cats.Eval
import com.thoughtworks.deepLearning.any.ast.Literal
import com.thoughtworks.deepLearning.double.ast.Add

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object double {

  type Double = {
    type Delta = Eval[scala.Double]
    type Data = Eval[scala.Double]
  }

  type DoubleBatch = Batch.Aux[Double#Data, Double#Delta]

  implicit final class DoubleOps[Input <: Batch](differentiable: Ast.Aux[Input, DoubleBatch]) {
    def +[RightInput <: Input](right: Ast.Aux[RightInput, DoubleBatch]) = {
      Add(differentiable, right)
    }
  }

  implicit def doubleLiteral[Input <: Batch](nativeDouble: scala.Double): Ast.Aux[Input, DoubleBatch] = {
    Literal(Eval.now(nativeDouble))
  }

}
