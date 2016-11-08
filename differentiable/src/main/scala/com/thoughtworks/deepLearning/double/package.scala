package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.any.Any
import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.Differentiable._

import scala.language.implicitConversions
import cats.Eval
import com.thoughtworks.deepLearning.any.ast.{Identity, Literal}
import com.thoughtworks.deepLearning.boolean.ast.If
import com.thoughtworks.deepLearning.double.ast._
import com.thoughtworks.deepLearning.boolean.utilities.Boolean

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object double {

  /** @template */
  type Double = utilities.Double

  implicit final class DoubleOps[Input <: Differentiable](differentiable: Ast[Input, Double#Widen]) {
    def +[RightInput <: Input](right: Ast[RightInput, Double#Widen]): Ast[RightInput, Double#Widen] = {
      Add(differentiable, right)
    }
    def -[RightInput <: Input](right: Ast[RightInput, Double#Widen]): Ast[RightInput, Double#Widen] = {
      Add(differentiable, Negative(right))
    }
    def /[RightInput <: Input](right: Ast[RightInput, Double#Widen]): Ast[RightInput, Double#Widen] = {
      Multiply(differentiable, Reciprocal(right))
    }
    def *[RightInput <: Input](right: Ast[RightInput, Double#Widen]): Ast[RightInput, Double#Widen] = {
      Multiply(differentiable, right)
    }
    def <[RightInput <: Input](right: Ast[RightInput, Double#Widen]): Ast[RightInput, Boolean#Widen] = {
      LessThan(differentiable, right)
    }
    def unary_- : Ast[Input, Double#Widen] = {
      Negative(differentiable)
    }

    def exp: Ast[Input, Double#Widen] = {
      Exp(differentiable)
    }

    def log: Ast[Input, Double#Widen] = {
      Log(differentiable)
    }

    def abs: Ast[Input, Double#Widen] = {
      If(differentiable < 0.0, -differentiable, differentiable)
    }

    def max[RightInput <: Input](
        rightAst: Ast[RightInput, Double#Widen]): Ast[RightInput, Double#Widen] = {
      If(differentiable < rightAst, rightAst, differentiable)
    }

    def min[RightInput <: Input](rightAst: Ast[RightInput, Double#Widen]): Ast[RightInput, Double#Widen] = {
      If(differentiable < rightAst, differentiable, rightAst)
    }

  }

  implicit def nativeDoubleToDoubleOps(nativeDouble: scala.Double): DoubleOps[Differentiable] = {
    DoubleOps(Literal(Eval.now(nativeDouble)))
  }

  implicit def doubleLiteral[Input <: Differentiable: Identity](nativeDouble: scala.Double): Ast[Input, Double#Widen] = {
    Literal(Eval.now(nativeDouble))
  }

  class InputTypePair[Data, Delta]

  implicit final class NativeDoubleOps(nativeDouble: scala.Double) {
    def toLiteral[Input <: Differentiable: Identity] = doubleLiteral(nativeDouble)
    def toWeight[Input <: Differentiable: Identity](implicit learningRate: LearningRate): Ast[Input, Double#Widen] = {
      Weight(nativeDouble)
    }
  }

}
