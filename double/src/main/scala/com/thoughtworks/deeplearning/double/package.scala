package com.thoughtworks.deeplearning
import cats.Eval
import com.thoughtworks.deeplearning.dsl._
import com.thoughtworks.deeplearning.Poly.PolyMethods._
import com.thoughtworks.deeplearning.Poly.PolyFunctions._
import com.thoughtworks.deeplearning.dsl.layers.Literal
import com.thoughtworks.deeplearning.BpBoolean.Layers.If
import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.Poly.PolyMethods
import com.thoughtworks.deeplearning.double.layers._
import com.thoughtworks.deeplearning.double.optimizers.Optimizer

import language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object double {

  /** @template */
  type BpDouble = com.thoughtworks.deeplearning.double.utilities.BpDouble

  implicit def liftNativeDoubleToLayer[InputData, InputDelta](implicit inputType: BackPropagationType[InputData, InputDelta])
    : ToLayer.Aux[Double, Batch.Aux[InputData, InputDelta], Eval[Double], Eval[Double]] =
    new ToLayer[Double, Batch.Aux[InputData, InputDelta]] {
      override type OutputData = Eval[Double]
      override type OutputDelta = Eval[Double]
      override def apply(nativeDouble: Double) = {
        Literal(Eval.now(nativeDouble))
      }
    }

  implicit def `min(Double,Double)`[Input <: Batch]
    : min.Case.Aux[Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch]] = {
    min.at { (leftLayer, rightLayer) =>
      If[Input, BpDouble#Data, BpDouble#Delta](LessThan[Input](leftLayer, rightLayer), leftLayer, rightLayer)
    }
  }
  implicit def `max(Double,Double)`[Input <: Batch]
    : max.Case.Aux[Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch]] = {
    max.at { (leftLayer, rightLayer) =>
      If[Input, BpDouble#Data, BpDouble#Delta](LessThan[Input](leftLayer, rightLayer), rightLayer, leftLayer)
    }
  }

  implicit def `Double-Double`[Input <: Batch]
    : -.Case.Aux[Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch]] = {
    PolyMethods.-.at { (leftLayer, rightLayer) =>
      Plus(leftLayer, Negative(rightLayer))
    }
  }

  implicit def `Double+Double`[Input <: Batch]
    : +.Case.Aux[Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch]] = {
    PolyMethods.+.at { (leftLayer, rightLayer) =>
      Plus(leftLayer, rightLayer)
    }
  }

  implicit def `Double/Double`[Input <: Batch]
    : /.Case.Aux[Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch]] = {
    /.at { (leftLayer, rightLayer) =>
      Times(leftLayer, Reciprocal(rightLayer))
    }
  }

  implicit def `Double*Double`[Input <: Batch]
    : *.Case.Aux[Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch]] = {
    *.at(Times(_, _))
  }

  implicit def `log(Double)`[Input <: Batch]
    : log.Case.Aux[Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch]] = {
    log.at(Log(_))
  }

  implicit def `exp(Double)`[Input <: Batch]
    : exp.Case.Aux[Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch]] = {
    exp.at(Exp(_))
  }

  implicit def `abs(Double)`[Input <: Batch]
    : abs.Case.Aux[Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch]] = {
    abs.at { operand =>
      If[Input, BpDouble#Data, BpDouble#Delta](LessThan(operand, Literal(Eval.now(0.0))), Negative(operand), operand)
    }
  }

  implicit final class NativeDoubleOps(nativeDouble: Double) {
    def toWeight[InputData, InputDelta](
                                         implicit inputType: BackPropagationType[InputData, InputDelta],
                                         optimizer: Optimizer): Layer.Aux[Batch.Aux[InputData, InputDelta], BpDouble#Batch] = {
      Weight(nativeDouble)
    }
  }

  final class DoubleOps[Input <: Batch](differentiable: Layer.Aux[Input, BpDouble#Batch]) {

    def unary_- : Layer.Aux[Input, BpDouble#Batch] = {
      Negative(differentiable)
    }

  }

  implicit def toDoubleOps[From, Input <: Batch](from: From)(
      implicit toLayer: ToLayer.OfType[From, Input, BpDouble]
  ): DoubleOps[Input] = {
    new DoubleOps(toLayer(from))
  }
}
