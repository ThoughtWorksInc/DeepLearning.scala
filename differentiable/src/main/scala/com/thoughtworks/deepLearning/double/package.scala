package com.thoughtworks.deepLearning
import cats.Eval
import com.thoughtworks.deepLearning.DifferentiableFunction.Ast
import com.thoughtworks.deepLearning.ToAst.InvariantAst
import com.thoughtworks.deepLearning.any.ast.Literal
import com.thoughtworks.deepLearning.boolean.ast.If
import com.thoughtworks.deepLearning.double.ast.LessThan

//import com.thoughtworks.deepLearning.any.Any
//import com.thoughtworks.deepLearning.DifferentiableFunction._
//import com.thoughtworks.deepLearning.Differentiable._
//import com.thoughtworks.Extractor._
//
//import scala.language.implicitConversions
//import cats.Eval
//import com.thoughtworks.deepLearning.Poly.{Poly1, Poly2}
//import com.thoughtworks.deepLearning.any.ast.{Identity, Literal}
//import com.thoughtworks.deepLearning.boolean.ast.If
//import com.thoughtworks.deepLearning.double.ast._
//import com.thoughtworks.deepLearning.double.utilities.Double
//import com.thoughtworks.deepLearning.boolean.utilities.Boolean

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object double {

  /** @template */
  type Double = utilities.Double

  implicit def liftNativeDouble[Input <: Differentiable: DifferentiableType.OfBatch]
    : ToAst.Aux[scala.Double, Input, Eval[scala.Double], Eval[scala.Double]] =
    new ToAst[scala.Double, Input] {
      override type OutputData = Eval[scala.Double]
      override type OutputDelta = Eval[scala.Double]
      override def apply(nativeDouble: scala.Double) = {
        Literal(Eval.now(nativeDouble))
      }
    }

  implicit def maxDoubleDouble[Left, Right, Input <: Differentiable](
      implicit leftToAst: ToAst.OfType[Left, Input, Double],
      rightToAst: ToAst.OfType[Right, Input, Double]): max.Case.Aux[Left, Right, Ast[Input, Double#Batch]] =
    max.at { (left, right) =>
      val leftAst = leftToAst(left)
      val rightAst = rightToAst(right)
      If[Input, Double#Batch](LessThan[Input](leftAst, rightAst), rightAst, leftAst)
    }
//
//  implicit final class DoubleOps[Input <: Differentiable](double: Ast[Input, Double#ConcreteBatch]) {
//    def +(right: Ast[Input, Double#ConcreteBatch]): Ast[Input, Double#ConcreteBatch] = {
//      Add(double, right)
//    }
//
//    def -(right: Ast[Input, Double#ConcreteBatch]): Ast[Input, Double#ConcreteBatch] = {
//      Add(double, Negative(right))
//    }
//
//    def /(right: Ast[Input, Double#ConcreteBatch]): Ast[Input, Double#ConcreteBatch] = {
//      Multiply(double, Reciprocal(right))
//    }
//
//    def *(right: Ast[Input, Double#ConcreteBatch]): Ast[Input, Double#ConcreteBatch] = {
//      Multiply(double, right)
//    }
//
//    def <(right: Ast[Input, Double#ConcreteBatch]): Ast[Input, Boolean#ConcreteBatch] = {
//      LessThan(double, right)
//    }
//
//    def unary_- : Ast[Input, Double#ConcreteBatch] = {
//      Negative(double)
//    }
//
//    def exp: Ast[Input, Double#ConcreteBatch] = {
//      Exp(double)
//    }
//
//    def log: Ast[Input, Double#ConcreteBatch] = {
//      Log(double)
//    }
//
//    def min(rightAst: Ast[Input, Double#ConcreteBatch]): Ast[Input, Double#ConcreteBatch] = {
//      If(LessThan(double, rightAst), double, rightAst)
//    }
//
//  }
//
//  implicit def nativeDoubleToAst[Input <: Differentiable: Identity] =
//    new ToAst[scala.Double, Input, Eval[scala.Double], Eval[scala.Double]] {
//      override def apply(nativeDouble: scala.Double) = Literal(Eval.now(nativeDouble))
//    }
//
//  implicit def maxDoubleDouble[Input <: Differentiable,
//                               LeftData <: Eval[scala.Double],
//                               LeftDelta >: Eval[scala.Double],
//                               RightData <: Eval[scala.Double],
//                               RightDelta >: Eval[scala.Double]]
//    : max.Case.Aux[Input, LeftData, LeftDelta, RightData, RightDelta, Ast[Input, Double#ConcreteBatch]] =
//    new max.Case[Input, LeftData, LeftDelta, RightData, RightDelta] {
//      override type Out = Ast[Input, Double#ConcreteBatch]
//      override def apply(leftOperand: Ast[Input, ConcreteBatch[LeftData, LeftDelta]],
//                         rightOperand: Ast[Input, ConcreteBatch[RightData, RightDelta]]) = {
//        If(LessThan(leftOperand, rightOperand), rightOperand, leftOperand)
//      }
//    }
//
//  implicit def absDouble[Input <: Differentiable, Data <: Eval[scala.Double], Delta >: Eval[scala.Double]]
//    : abs.Case[Input, Data, Delta] {
//      type Out = Ast[Input, Double#ConcreteBatch]
//    } =
//    new abs.Case[Input, Data, Delta] {
//      override type Out = Ast[Input, Double#ConcreteBatch]
//      override def apply(operand: Ast[Input, ConcreteBatch[Data, Delta]]) = {
//        If(LessThan(operand, 0.0), Negative(operand), operand)
//      }
//    }
//
//  implicit def nativeDoubleToDoubleOps(nativeDouble: scala.Double): DoubleOps[Differentiable] = {
//    DoubleOps(Literal(Eval.now(nativeDouble)))
//  }
//
//  implicit def doubleLiteral[Input <: Differentiable: Identity](
//      nativeDouble: scala.Double): Ast[Input, ConcreteBatch[Double#Data, Double#Delta]] = {
//    Literal(Eval.now(nativeDouble))
//  }
//
//  implicit final class NativeDoubleOps(nativeDouble: scala.Double) {
//    def toLiteral[Input <: Differentiable: Identity] = doubleLiteral(nativeDouble)
//
//    def toWeight[Input <: Differentiable: Identity](implicit learningRate: LearningRate): Ast[Input, Double#ConcreteBatch] = {
//      Weight(nativeDouble)
//    }
//  }
//
}
