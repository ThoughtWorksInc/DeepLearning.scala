package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.any.Any
import com.thoughtworks.deepLearning.Ast._
import com.thoughtworks.deepLearning.Batch._
import scala.language.implicitConversions
import cats.Eval
import com.thoughtworks.deepLearning.any.ast.{Identity, Literal}
import com.thoughtworks.deepLearning.double.ast.{Add, Weight}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object double {

  /** @template */
  type Double = utilities.Double

  implicit final class DoubleOps[Input <: Batch](differentiable: WidenAst[Input, Double#Widen]) {
    def +[RightInput <: Input](right: WidenAst[RightInput, Double#Widen]) = {
      Add(differentiable, right)
    }
  }

  implicit def doubleLiteral[Input <: Batch: Identity](nativeDouble: scala.Double): WidenAst[Input, Double#Widen] = {
    Literal(Eval.now(nativeDouble))
  }

  class InputTypePair[Data, Delta]

  implicit final class NativeDoubleOps(nativeDouble: scala.Double) {
    def toLiteral[Input <: Batch: Identity] = doubleLiteral(nativeDouble)
    def toWeight[Input <: Batch: Identity](implicit learningRate: LearningRate): WidenAst[Input, Double#Widen] = {
      Weight(nativeDouble)
    }
  }

}
