package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.any.Any
import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.Differentiable._
import com.thoughtworks.Extractor._

import scala.language.implicitConversions
import cats.Eval
import com.thoughtworks.deepLearning.any.ast.{Identity, Literal}
import com.thoughtworks.deepLearning.boolean.ast.If
import com.thoughtworks.deepLearning.double.ast._
import com.thoughtworks.deepLearning.double.utilities.Double
import com.thoughtworks.deepLearning.boolean.utilities.Boolean

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object double {

  /** @template */
  type Double = utilities.Double

  implicit final class DoubleOps[Input <: Differentiable](double: DifferentiableFunction.Ast[Input, Double#Batch]) {
    def +[RightInput <: Input](right: DifferentiableFunction.Ast[RightInput, Double#Batch])
      : DifferentiableFunction.Ast[RightInput, Double#Batch] = {
      Add(double, right)
    }

    def -[RightInput <: Input](right: DifferentiableFunction.Ast[RightInput, Double#Batch])
      : DifferentiableFunction.Ast[RightInput, Double#Batch] = {
      Add(double, Negative(right))
    }

    def /[RightInput <: Input](right: DifferentiableFunction.Ast[RightInput, Double#Batch])
      : DifferentiableFunction.Ast[RightInput, Double#Batch] = {
      Multiply(double, Reciprocal(right))
    }

    def *[RightInput <: Input](right: DifferentiableFunction.Ast[RightInput, Double#Batch])
      : DifferentiableFunction.Ast[RightInput, Double#Batch] = {
      Multiply(double, right)
    }

    def <[RightInput <: Input](right: DifferentiableFunction.Ast[RightInput, Double#Batch])
      : DifferentiableFunction.Ast[RightInput, Boolean#Batch] = {
      LessThan(double, right)
    }

    def unary_- : DifferentiableFunction.Ast[Input, Double#Batch] = {
      Negative(double)
    }

    def exp: DifferentiableFunction.Ast[Input, Double#Batch] = {
      Exp(double)
    }

    def log: DifferentiableFunction.Ast[Input, Double#Batch] = {
      Log(double)
    }

    def min[RightInput <: Input](rightAst: DifferentiableFunction.Ast[RightInput, Double#Batch])
      : DifferentiableFunction.Ast[RightInput, Double#Batch] = {
      If(double < rightAst, double, rightAst)
    }

  }

  implicit def maxDoubleDouble[Input <: Differentiable, Left, Right](
      implicit leftView: ToAst[Left, Input, Eval[scala.Double], Eval[scala.Double]],
      rightView: ToAst[Right, Input, Eval[scala.Double], Eval[scala.Double]])
    : max.Case.Aux[Left, Right, DifferentiableFunction.Ast[Input, Double#Batch]] =
    max.at { (left, right) =>
      val leftAst = leftView(left)
      val rightAst = rightView(right)
      If(leftAst < rightAst, rightAst, leftAst)
    }

  implicit def absDouble[Input <: Differentiable: Identity]
    : abs.Case.Aux[Ast[Input, Double#Batch], Ast[Input, Double#Batch]] =
    abs.at { operand =>
      If(operand < 0.0, -operand, operand)
    }

  implicit def nativeDoubleToDoubleOps(nativeDouble: scala.Double): DoubleOps[Differentiable] = {
    DoubleOps(Literal(Eval.now(nativeDouble)))
  }

  implicit def doubleLiteral[Input <: Differentiable: Identity](
      nativeDouble: scala.Double): DifferentiableFunction.Ast[Input, Double#Batch] = {
    Literal(Eval.now(nativeDouble))
  }

  implicit final class NativeDoubleOps(nativeDouble: scala.Double) {
    def toLiteral[Input <: Differentiable: Identity] = doubleLiteral(nativeDouble)

    def toWeight[Input <: Differentiable: Identity](
        implicit learningRate: LearningRate): DifferentiableFunction.Ast[Input, Double#Batch] = {
      Weight(nativeDouble)
    }
  }

}
