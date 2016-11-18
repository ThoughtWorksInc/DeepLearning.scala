package com.thoughtworks.deepLearning
import cats.Eval

import com.thoughtworks.deepLearning.any._
import com.thoughtworks.deepLearning.any.AstMethods._
import com.thoughtworks.deepLearning.any.ast.Literal
import com.thoughtworks.deepLearning.boolean.ast.If
import com.thoughtworks.deepLearning.double.ast._

import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object double {

  /** @template */
  type Double = utilities.Double

  implicit def liftNativeDoubleToNeuralNetwork[InputData, InputDelta](implicit inputType: Type[InputData, InputDelta])
    : ToNeuralNetwork.Aux[scala.Double, Batch.Aux[InputData, InputDelta], Eval[scala.Double], Eval[scala.Double]] =
    new ToNeuralNetwork[scala.Double, Batch.Aux[InputData, InputDelta]] {
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
      If[Input, Double#Data, Double#Delta](LessThan[Input](leftAst, rightAst), leftAst, rightAst)
    }
  }
  implicit def `max(Double,Double)`[Input <: Batch]: max.Case.Aux[NeuralNetwork.Aux[Input, Double#Batch],
                                                                  NeuralNetwork.Aux[Input, Double#Batch],
                                                                  NeuralNetwork.Aux[Input, Double#Batch]] = {
    max.at { (leftAst, rightAst) =>
      If[Input, Double#Data, Double#Delta](LessThan[Input](leftAst, rightAst), rightAst, leftAst)
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
      Plus(leftAst, rightAst)
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

  implicit def `exp(Double)`[Input <: Batch]
    : exp.Case.Aux[NeuralNetwork.Aux[Input, Double#Batch], NeuralNetwork.Aux[Input, Double#Batch]] = {
    exp.at(Exp(_))
  }

  implicit def `abs(Double)`[Input <: Batch]
    : abs.Case.Aux[NeuralNetwork.Aux[Input, Double#Batch], NeuralNetwork.Aux[Input, Double#Batch]] = {
    abs.at { operand =>
      If[Input, Double#Data, Double#Delta](LessThan(operand, Literal(Eval.now(0.0))), Negative(operand), operand)
    }
  }

  implicit final class NativeDoubleOps(nativeDouble: scala.Double) {
    def toWeight[InputData, InputDelta](
        implicit inputType: Type[InputData, InputDelta],
        learningRate: LearningRate): NeuralNetwork.Aux[Batch.Aux[InputData, InputDelta], Double#Batch] = {
      Weight(nativeDouble)
    }
  }

  final class DoubleOps[Input <: Batch](differentiable: NeuralNetwork.Aux[Input, Double#Batch]) {

    def unary_- : NeuralNetwork.Aux[Input, Double#Batch] = {
      Negative(differentiable)
    }

  }

  implicit def toDoubleOps[From, Input <: Batch](from: From)(
      implicit toNeuralNetwork: ToNeuralNetwork.OfType[From, Input, Double]
  ): DoubleOps[Input] = {
    new DoubleOps(toNeuralNetwork(from))
  }
}
