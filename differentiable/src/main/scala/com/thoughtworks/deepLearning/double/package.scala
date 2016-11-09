package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.any.Any
import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.Differentiable._
import com.thoughtworks.Extractor._

import scala.language.implicitConversions
import cats.Eval
import com.thoughtworks.deepLearning.Poly.{Poly1, Poly2}
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

  implicit def nativeDoubleToAst[Input <: Differentiable: Identity] =
    new ToAst[scala.Double, Input, Eval[scala.Double], Eval[scala.Double]] {
      override def apply(nativeDouble: scala.Double) = Literal(Eval.now(nativeDouble))
    }

  implicit def maxDoubleDouble[Input <: Differentiable] =
    new max.Case[Input, Eval[scala.Double], Eval[scala.Double], Eval[scala.Double], Eval[scala.Double]] {
      override type Out = Ast[Input, Double#Batch]
      override def apply(leftOperand: Ast[Input, Batch[cats.Eval[scala.Double], cats.Eval[scala.Double]]],
                         rightOperand: Ast[Input, Batch[cats.Eval[scala.Double], cats.Eval[scala.Double]]]) = {
        If(leftOperand < rightOperand, rightOperand, leftOperand)
      }
    }

  implicit def absDouble[Input <: Differentiable]: abs.Case[Input, Eval[scala.Double], Eval[scala.Double]] {
    type Out = Ast[Input, Double#Batch]
  } =
    new abs.Case[Input, Eval[scala.Double], Eval[scala.Double]] {
      override type Out = Ast[Input, Double#Batch]
      override def apply(operand: Ast[Input, Batch[Eval[scala.Double], Eval[scala.Double]]]) = {
        If(operand < 0.0, -operand, operand)
      }
    }

  implicit def nativeDoubleToDoubleOps(nativeDouble: scala.Double): DoubleOps[Differentiable] = {
    DoubleOps(Literal(Eval.now(nativeDouble)))
  }

  implicit def doubleLiteral[Input <: Differentiable: Identity](
      nativeDouble: scala.Double): DifferentiableFunction.Ast[Input, Batch[Double#Data, Double#Delta]] = {
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
