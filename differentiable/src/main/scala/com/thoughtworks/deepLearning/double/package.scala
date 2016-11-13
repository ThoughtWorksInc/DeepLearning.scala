package com.thoughtworks.deepLearning
import cats.Eval
import com.thoughtworks.deepLearning.Differentiable.Batch
import com.thoughtworks.deepLearning.DifferentiableFunction.Ast
import AstMethods._
import com.thoughtworks.deepLearning.any.ast.Literal
import com.thoughtworks.deepLearning.boolean.ast.If
import com.thoughtworks.deepLearning.double.ast._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object double {

  /** @template */
  type Double = utilities.Double
  val Double: Double = implicitly

  implicit def liftNativeDoubleToAst[InputData, InputDelta](
      implicit inputType: DifferentiableType[InputData, InputDelta])
    : ToAst.Aux[scala.Double, Batch[InputData, InputDelta], Eval[scala.Double], Eval[scala.Double]] =
    new ToAst[scala.Double, Batch[InputData, InputDelta]] {
      override type OutputData = Eval[scala.Double]
      override type OutputDelta = Eval[scala.Double]
      override def apply(nativeDouble: scala.Double) = {
        Literal(Eval.now(nativeDouble))
      }
    }

  implicit def `max(Double,Double)`[Input <: Differentiable]
    : max.Case.Aux[Ast[Input, Double#Batch], Ast[Input, Double#Batch], Ast[Input, Double#Batch]] = {
    max.at { (leftAst, rightAst) =>
      If[Input, Double#Batch](LessThan[Input](leftAst, rightAst), rightAst, leftAst)
    }
  }

  implicit def `Double-Double`[Input <: Differentiable]
    : -.Case.Aux[Ast[Input, Double#Batch], Ast[Input, Double#Batch], Ast[Input, Double#Batch]] = {
    AstMethods.-.at { (leftAst, rightAst) =>
      Plus(leftAst, Negative(rightAst))
    }
  }

  implicit def `Double+Double`[Input <: Differentiable]
    : +.Case.Aux[Ast[Input, Double#Batch], Ast[Input, Double#Batch], Ast[Input, Double#Batch]] = {
    AstMethods.+.at { (leftAst, rightAst) =>
      Plus(leftAst, Negative(rightAst))
    }
  }

  implicit def `Double/Double`[Input <: Differentiable]
    : /.Case.Aux[Ast[Input, Double#Batch], Ast[Input, Double#Batch], Ast[Input, Double#Batch]] = {
    /.at { (leftAst, rightAst) =>
      Times(leftAst, Reciprocal(rightAst))
    }
  }

  implicit def `Double*Double`[Input <: Differentiable]
    : *.Case.Aux[Ast[Input, Double#Batch], Ast[Input, Double#Batch], Ast[Input, Double#Batch]] = {
    *.at(Times(_, _))
  }

  implicit def `log(Double)`[Input <: Differentiable]
    : log.Case.Aux[Ast[Input, Double#Batch], Ast[Input, Double#Batch]] = {
    log.at(Log(_))
  }
  implicit def `abs(Double)`[Input <: Differentiable]
    : abs.Case.Aux[Ast[Input, Double#Batch], Ast[Input, Double#Batch]] = {
    abs.at { operand =>
      If(LessThan(operand, Literal(Eval.now(0.0))), Negative(operand), operand)
    }
  }
//
//  implicit final class DoubleOps[Input <: Differentiable](double: Ast[Input, Double#ConcreteBatch]) {
//    def +(right: Ast[Input, Double#ConcreteBatch]): Ast[Input, Double#ConcreteBatch] = {
//      Plus(double, right)
//    }
//
//    def -(right: Ast[Input, Double#ConcreteBatch]): Ast[Input, Double#ConcreteBatch] = {
//      Plus(double, Negative(right))
//    }
//
//    def /(right: Ast[Input, Double#ConcreteBatch]): Ast[Input, Double#ConcreteBatch] = {
//      Times(double, Reciprocal(right))
//    }
//
//    def *(right: Ast[Input, Double#ConcreteBatch]): Ast[Input, Double#ConcreteBatch] = {
//      Times(double, right)
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
