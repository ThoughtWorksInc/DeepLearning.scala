package com.thoughtworks.deepLearning
import cats.Eval
import com.thoughtworks.deepLearning.Batch.Aux
import com.thoughtworks.deepLearning.NeuralNetwork.Aux
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

  implicit def liftNativeDoubleToNeuralNetwork[InputData, InputDelta](implicit inputType: Type[InputData, InputDelta])
    : IsNeuralNetwork.Aux[scala.Double, Batch.Aux[InputData, InputDelta], Eval[scala.Double], Eval[scala.Double]] =
    new IsNeuralNetwork[scala.Double, Batch.Aux[InputData, InputDelta]] {
      override type OutputData = Eval[scala.Double]
      override type OutputDelta = Eval[scala.Double]
      override def apply(nativeDouble: scala.Double) = {
        Literal(Eval.now(nativeDouble))
      }
    }

  implicit def `min(Double,Double)`[Input <: Batch]: min.Case.Aux[NeuralNetwork.Aux[Input, Double#Batch],
                                                                  NeuralNetwork.Aux[Input, Double#Batch],
                                                                  NeuralNetwork.Aux[Input, Double#Batch]] = {
    min.at { (leftAst, rightAst) =>
      If[Input, Double#Batch](LessThan[Input](leftAst, rightAst), leftAst, rightAst)
    }
  }
  implicit def `max(Double,Double)`[Input <: Batch]: max.Case.Aux[NeuralNetwork.Aux[Input, Double#Batch],
                                                                  NeuralNetwork.Aux[Input, Double#Batch],
                                                                  NeuralNetwork.Aux[Input, Double#Batch]] = {
    max.at { (leftAst, rightAst) =>
      If[Input, Double#Batch](LessThan[Input](leftAst, rightAst), rightAst, leftAst)
    }
  }

  implicit def `Double-Double`[Input <: Batch]: -.Case.Aux[NeuralNetwork.Aux[Input, Double#Batch],
                                                           NeuralNetwork.Aux[Input, Double#Batch],
                                                           NeuralNetwork.Aux[Input, Double#Batch]] = {
    AstMethods.-.at { (leftAst, rightAst) =>
      Plus(leftAst, Negative(rightAst))
    }
  }

  implicit def `Double+Double`[Input <: Batch]: +.Case.Aux[NeuralNetwork.Aux[Input, Double#Batch],
                                                           NeuralNetwork.Aux[Input, Double#Batch],
                                                           NeuralNetwork.Aux[Input, Double#Batch]] = {
    AstMethods.+.at { (leftAst, rightAst) =>
      Plus(leftAst, Negative(rightAst))
    }
  }

  implicit def `Double/Double`[Input <: Batch]: /.Case.Aux[NeuralNetwork.Aux[Input, Double#Batch],
                                                           NeuralNetwork.Aux[Input, Double#Batch],
                                                           NeuralNetwork.Aux[Input, Double#Batch]] = {
    /.at { (leftAst, rightAst) =>
      Times(leftAst, Reciprocal(rightAst))
    }
  }

  implicit def `Double*Double`[Input <: Batch]: *.Case.Aux[NeuralNetwork.Aux[Input, Double#Batch],
                                                           NeuralNetwork.Aux[Input, Double#Batch],
                                                           NeuralNetwork.Aux[Input, Double#Batch]] = {
    *.at(Times(_, _))
  }

  implicit def `log(Double)`[Input <: Batch]
    : log.Case.Aux[NeuralNetwork.Aux[Input, Double#Batch], NeuralNetwork.Aux[Input, Double#Batch]] = {
    log.at(Log(_))
  }
  implicit def `abs(Double)`[Input <: Batch]
    : abs.Case.Aux[NeuralNetwork.Aux[Input, Double#Batch], NeuralNetwork.Aux[Input, Double#Batch]] = {
    abs.at { operand =>
      If(LessThan(operand, Literal(Eval.now(0.0))), Negative(operand), operand)
    }
  }
//
//  implicit final class DoubleOps[Input <: Batch](double: NeuralNetwork.Aux[Input, Double#ConcreteBatch]) {
//    def +(right: NeuralNetwork.Aux[Input, Double#ConcreteBatch]): NeuralNetwork.Aux[Input, Double#ConcreteBatch] = {
//      Plus(double, right)
//    }
//
//    def -(right: NeuralNetwork.Aux[Input, Double#ConcreteBatch]): NeuralNetwork.Aux[Input, Double#ConcreteBatch] = {
//      Plus(double, Negative(right))
//    }
//
//    def /(right: NeuralNetwork.Aux[Input, Double#ConcreteBatch]): NeuralNetwork.Aux[Input, Double#ConcreteBatch] = {
//      Times(double, Reciprocal(right))
//    }
//
//    def *(right: NeuralNetwork.Aux[Input, Double#ConcreteBatch]): NeuralNetwork.Aux[Input, Double#ConcreteBatch] = {
//      Times(double, right)
//    }
//
//    def <(right: NeuralNetwork.Aux[Input, Double#ConcreteBatch]): NeuralNetwork.Aux[Input, Boolean#ConcreteBatch] = {
//      LessThan(double, right)
//    }
//
//    def unary_- : NeuralNetwork.Aux[Input, Double#ConcreteBatch] = {
//      Negative(double)
//    }
//
//    def exp: NeuralNetwork.Aux[Input, Double#ConcreteBatch] = {
//      Exp(double)
//    }
//
//    def log: NeuralNetwork.Aux[Input, Double#ConcreteBatch] = {
//      Log(double)
//    }
//
//    def min(rightAst: NeuralNetwork.Aux[Input, Double#ConcreteBatch]): NeuralNetwork.Aux[Input, Double#ConcreteBatch] = {
//      If(LessThan(double, rightAst), double, rightAst)
//    }
//
//  }
//
//  implicit def nativeDoubleIsNeuralNetwork[Input <: Batch: Identity] =
//    new IsNeuralNetwork[scala.Double, Input, Eval[scala.Double], Eval[scala.Double]] {
//      override def apply(nativeDouble: scala.Double) = Literal(Eval.now(nativeDouble))
//    }
//
//  implicit def maxDoubleDouble[Input <: Batch,
//                               LeftData <: Eval[scala.Double],
//                               LeftDelta >: Eval[scala.Double],
//                               RightData <: Eval[scala.Double],
//                               RightDelta >: Eval[scala.Double]]
//    : max.Case.Aux[Input, LeftData, LeftDelta, RightData, RightDelta, NeuralNetwork.Aux[Input, Double#ConcreteBatch]] =
//    new max.Case[Input, LeftData, LeftDelta, RightData, RightDelta] {
//      override type Out = NeuralNetwork.Aux[Input, Double#ConcreteBatch]
//      override def apply(leftOperand: NeuralNetwork.Aux[Input, ConcreteBatch[LeftData, LeftDelta]],
//                         rightOperand: NeuralNetwork.Aux[Input, ConcreteBatch[RightData, RightDelta]]) = {
//        If(LessThan(leftOperand, rightOperand), rightOperand, leftOperand)
//      }
//    }
//
//  implicit def absDouble[Input <: Batch, Data <: Eval[scala.Double], Delta >: Eval[scala.Double]]
//    : abs.Case[Input, Data, Delta] {
//      type Out = NeuralNetwork.Aux[Input, Double#ConcreteBatch]
//    } =
//    new abs.Case[Input, Data, Delta] {
//      override type Out = NeuralNetwork.Aux[Input, Double#ConcreteBatch]
//      override def apply(operand: NeuralNetwork.Aux[Input, ConcreteBatch[Data, Delta]]) = {
//        If(LessThan(operand, 0.0), Negative(operand), operand)
//      }
//    }
//
//  implicit def nativeDoubleToDoubleOps(nativeDouble: scala.Double): DoubleOps[Batch] = {
//    DoubleOps(Literal(Eval.now(nativeDouble)))
//  }
//
//  implicit def doubleLiteral[Input <: Batch: Identity](
//      nativeDouble: scala.Double): NeuralNetwork.Aux[Input, ConcreteBatch[Double#Data, Double#Delta]] = {
//    Literal(Eval.now(nativeDouble))
//  }
//
//  implicit final class NativeDoubleOps(nativeDouble: scala.Double) {
//    def toLiteral[Input <: Batch: Identity] = doubleLiteral(nativeDouble)
//
//    def toWeight[Input <: Batch: Identity](implicit learningRate: LearningRate): NeuralNetwork.Aux[Input, Double#ConcreteBatch] = {
//      Weight(nativeDouble)
//    }
//  }
//
}
