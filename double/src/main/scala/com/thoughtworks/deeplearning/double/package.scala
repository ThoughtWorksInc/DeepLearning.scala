package com.thoughtworks.deeplearning
import cats.Eval
import com.thoughtworks.deeplearning.any._
import com.thoughtworks.deeplearning.any.PolyMethods._
import com.thoughtworks.deeplearning.any.layers.Literal
import com.thoughtworks.deeplearning.boolean.layers.If
import com.thoughtworks.deeplearning.double.layers._
import com.thoughtworks.deeplearning.double.optimizers.Optimizer

import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object double {

  /** @template */
  type Double = utilities.Double

  implicit def liftNativeDoubleToLayer[InputData, InputDelta](implicit inputType: Type[InputData, InputDelta])
    : ToLayer.Aux[scala.Double, Batch.Aux[InputData, InputDelta], Eval[scala.Double], Eval[scala.Double]] =
    new ToLayer[scala.Double, Batch.Aux[InputData, InputDelta]] {
      override type OutputData = Eval[scala.Double]
      override type OutputDelta = Eval[scala.Double]
      override def apply(nativeDouble: scala.Double) = {
        Literal(Eval.now(nativeDouble))
      }
    }

  implicit def `min(Double,Double)`[Input <: Batch]
    : min.Case.Aux[Layer.Aux[Input, Double#Batch], Layer.Aux[Input, Double#Batch], Layer.Aux[Input, Double#Batch]] = {
    min.at { (leftLayer, rightLayer) =>
      If[Input, Double#Data, Double#Delta](LessThan[Input](leftLayer, rightLayer), leftLayer, rightLayer)
    }
  }
  implicit def `max(Double,Double)`[Input <: Batch]
    : max.Case.Aux[Layer.Aux[Input, Double#Batch], Layer.Aux[Input, Double#Batch], Layer.Aux[Input, Double#Batch]] = {
    max.at { (leftLayer, rightLayer) =>
      If[Input, Double#Data, Double#Delta](LessThan[Input](leftLayer, rightLayer), rightLayer, leftLayer)
    }
  }

  implicit def `Double-Double`[Input <: Batch]
    : -.Case.Aux[Layer.Aux[Input, Double#Batch], Layer.Aux[Input, Double#Batch], Layer.Aux[Input, Double#Batch]] = {
    PolyMethods.-.at { (leftLayer, rightLayer) =>
      Plus(leftLayer, Negative(rightLayer))
    }
  }

  implicit def `Double+Double`[Input <: Batch]
    : +.Case.Aux[Layer.Aux[Input, Double#Batch], Layer.Aux[Input, Double#Batch], Layer.Aux[Input, Double#Batch]] = {
    PolyMethods.+.at { (leftLayer, rightLayer) =>
      Plus(leftLayer, rightLayer)
    }
  }

  implicit def `Double/Double`[Input <: Batch]
    : /.Case.Aux[Layer.Aux[Input, Double#Batch], Layer.Aux[Input, Double#Batch], Layer.Aux[Input, Double#Batch]] = {
    /.at { (leftLayer, rightLayer) =>
      Times(leftLayer, Reciprocal(rightLayer))
    }
  }

  implicit def `Double*Double`[Input <: Batch]
    : *.Case.Aux[Layer.Aux[Input, Double#Batch], Layer.Aux[Input, Double#Batch], Layer.Aux[Input, Double#Batch]] = {
    *.at(Times(_, _))
  }

  implicit def `log(Double)`[Input <: Batch]
    : log.Case.Aux[Layer.Aux[Input, Double#Batch], Layer.Aux[Input, Double#Batch]] = {
    log.at(Log(_))
  }

  implicit def `exp(Double)`[Input <: Batch]
    : exp.Case.Aux[Layer.Aux[Input, Double#Batch], Layer.Aux[Input, Double#Batch]] = {
    exp.at(Exp(_))
  }

  implicit def `abs(Double)`[Input <: Batch]
    : abs.Case.Aux[Layer.Aux[Input, Double#Batch], Layer.Aux[Input, Double#Batch]] = {
    abs.at { operand =>
      If[Input, Double#Data, Double#Delta](LessThan(operand, Literal(Eval.now(0.0))), Negative(operand), operand)
    }
  }

  implicit final class NativeDoubleOps(nativeDouble: scala.Double) {
    def toWeight[InputData, InputDelta](
        implicit inputType: Type[InputData, InputDelta],
        optimizer: Optimizer): Layer.Aux[Batch.Aux[InputData, InputDelta], Double#Batch] = {
      Weight(nativeDouble)
    }
  }

  final class DoubleOps[Input <: Batch](differentiable: Layer.Aux[Input, Double#Batch]) {

    def unary_- : Layer.Aux[Input, Double#Batch] = {
      Negative(differentiable)
    }

  }

  implicit def toDoubleOps[From, Input <: Batch](from: From)(
      implicit toLayer: ToLayer.OfType[From, Input, Double]
  ): DoubleOps[Input] = {
    new DoubleOps(toLayer(from))
  }
}
