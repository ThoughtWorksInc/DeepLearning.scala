package com.thoughtworks.deepLearning

import scala.language.implicitConversions
import cats.Eval
import com.thoughtworks.deepLearning.any.ast.{Identity, Literal}
import com.thoughtworks.deepLearning.double.ast.{Add, Weight}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object double {

  type Double = {
    type Delta = Eval[scala.Double]
    type Data = Eval[scala.Double]
  }

  implicit final class DoubleOps[Input <: Batch](
      differentiable: Ast.Aux[Input, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]) {
    def +[RightInput <: Input](right: Ast.Aux[RightInput, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]) = {
      Add(differentiable, right)
    }
  }

  implicit def doubleLiteral[Input <: Batch: Identity](
      nativeDouble: scala.Double): Ast.Aux[Input, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]] = {
    Literal(Eval.now(nativeDouble))
  }

  class InputTypePair[Data, Delta]

  implicit final class NativeDoubleOps(nativeDouble: scala.Double) {
    def toLiteral[Input <: Batch: Identity] = doubleLiteral(nativeDouble)
    def toWeight[Input <: Batch: Identity](
        implicit learningRate: LearningRate): Ast.Aux[Input, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]] = {
      Weight(nativeDouble)
    }
  }

}
