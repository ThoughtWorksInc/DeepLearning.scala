package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.any.Any
import com.thoughtworks.deepLearning.core.DifferentiableFunction._
import com.thoughtworks.deepLearning.core.Differentiable._

import scala.language.implicitConversions
import cats.Eval
import com.thoughtworks.deepLearning.any.ast.{Identity, Literal}
import com.thoughtworks.deepLearning.boolean.ast.If
import com.thoughtworks.deepLearning.double.ast._
import com.thoughtworks.deepLearning.boolean.utilities.Boolean
import com.thoughtworks.deepLearning.core.{Differentiable, DifferentiableFunction, LearningRate}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object double {

  /** @template */
  type Double = utilities.Double

  implicit final class DoubleOps[Input <: Differentiable](differentiable: DifferentiableFunction.Ast[Input, Double#Batch]) {
    def +[RightInput <: Input](right: DifferentiableFunction.Ast[RightInput, Double#Batch]): DifferentiableFunction.Ast[RightInput, Double#Batch] = {
      Add(differentiable, right)
    }
    def -[RightInput <: Input](right: DifferentiableFunction.Ast[RightInput, Double#Batch]): DifferentiableFunction.Ast[RightInput, Double#Batch] = {
      Add(differentiable, Negative(right))
    }
    def /[RightInput <: Input](right: DifferentiableFunction.Ast[RightInput, Double#Batch]): DifferentiableFunction.Ast[RightInput, Double#Batch] = {
      Multiply(differentiable, Reciprocal(right))
    }
    def *[RightInput <: Input](right: DifferentiableFunction.Ast[RightInput, Double#Batch]): DifferentiableFunction.Ast[RightInput, Double#Batch] = {
      Multiply(differentiable, right)
    }
    def <[RightInput <: Input](right: DifferentiableFunction.Ast[RightInput, Double#Batch]): DifferentiableFunction.Ast[RightInput, Boolean#Batch] = {
      LessThan(differentiable, right)
    }
    def unary_- : DifferentiableFunction.Ast[Input, Double#Batch] = {
      Negative(differentiable)
    }

    def exp: DifferentiableFunction.Ast[Input, Double#Batch] = {
      Exp(differentiable)
    }

    def log: DifferentiableFunction.Ast[Input, Double#Batch] = {
      Log(differentiable)
    }

    def abs: DifferentiableFunction.Ast[Input, Double#Batch] = {
      If(differentiable < 0.0, -differentiable, differentiable)
    }

    def max[RightInput <: Input](
        rightAst: DifferentiableFunction.Ast[RightInput, Double#Batch]): DifferentiableFunction.Ast[RightInput, Double#Batch] = {
      If(differentiable < rightAst, rightAst, differentiable)
    }

    def min[RightInput <: Input](rightAst: DifferentiableFunction.Ast[RightInput, Double#Batch]): DifferentiableFunction.Ast[RightInput, Double#Batch] = {
      If(differentiable < rightAst, differentiable, rightAst)
    }

  }

  implicit def nativeDoubleToDoubleOps(nativeDouble: scala.Double): DoubleOps[Differentiable] = {
    DoubleOps(Literal(Eval.now(nativeDouble)))
  }

  implicit def doubleLiteral[Input <: Differentiable: Identity](nativeDouble: scala.Double): DifferentiableFunction.Ast[Input, Double#Batch] = {
    Literal(Eval.now(nativeDouble))
  }

  class InputTypePair[Data, Delta]

  implicit final class NativeDoubleOps(nativeDouble: scala.Double) {
    def toLiteral[Input <: Differentiable: Identity] = doubleLiteral(nativeDouble)
    def toWeight[Input <: Differentiable: Identity](implicit learningRate: LearningRate): DifferentiableFunction.Ast[Input, Double#Batch] = {
      Weight(nativeDouble)
    }
  }

}
