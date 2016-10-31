package com.thoughtworks.deepLearning

import scala.language.implicitConversions
import cats.Eval
import com.thoughtworks.deepLearning.any.ast.Literal
import com.thoughtworks.deepLearning.double.ast.{Add, Weight}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object double {

  type Double = {
    type Delta = Eval[scala.Double]
    type Data = Eval[scala.Double]
  }

//  type DoubleBatch = Batch.Aux[Eval[scala.Double], Eval[scala.Double]]

  implicit final class DoubleOps[Input <: Batch](
      differentiable: Ast.Aux[Input, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]) {
    def +[RightInput <: Input](right: Ast.Aux[RightInput, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]) = {
      Add(differentiable, right)
    }
  }

  implicit def doubleLiteral[Input <: Batch](
      nativeDouble: scala.Double): Ast.Aux[Input, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]] = {
    Literal(Eval.now(nativeDouble))
  }

  def weight(nativeDouble: scala.Double)(implicit learningRate: LearningRate) = {
    Weight(nativeDouble)
  }

  def literal(nativeDouble: scala.Double) = {
    Literal(Eval.now(nativeDouble))
  }

}
