package com.thoughtworks.deeplearning

import cats.Eval
import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.dsl._
import com.thoughtworks.deeplearning.array2D.layers._
import com.thoughtworks.deeplearning.array2D.optimizers.Optimizer
import com.thoughtworks.deeplearning.double.utilities.BpDouble
import org.nd4j.linalg.api.ndarray.INDArray

import language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object array2D {

  /** @template */
  type Array2D = com.thoughtworks.deeplearning.array2D.utilities.Array2D

  implicit def `max(Array2D,Double)`[Left, Right, Input <: Batch]
    : max.Case.Aux[Layer.Aux[Input, Array2D#Batch], Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, Array2D#Batch]] =
    max.at(MaxDouble(_, _))

  implicit def `Array2D/Array2D`[Input <: Batch]: PolyMethods./.Case.Aux[Layer.Aux[Input, Array2D#Batch],
                                                                         Layer.Aux[Input, Array2D#Batch],
                                                                         Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods./.at { (leftLayer, rightLayer) =>
      MultiplyArray2D(leftLayer, Reciprocal(rightLayer))
    }
  }

  implicit def `Double/Array2D`[Input <: Batch]: PolyMethods./.Case.Aux[Layer.Aux[Input, BpDouble#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods./.at { (leftLayer, rightLayer) =>
      MultiplyDouble(Reciprocal(rightLayer), leftLayer)
    }
  }

  implicit def `Array2D/Double`[Input <: Batch]: PolyMethods./.Case.Aux[Layer.Aux[Input, Array2D#Batch],
                                                                        Layer.Aux[Input, BpDouble#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods./.at { (leftLayer, rightLayer) =>
      MultiplyDouble(leftLayer, double.layers.Reciprocal(rightLayer))
    }
  }

  implicit def `Array2D*Array2D`[Input <: Batch]: PolyMethods.*.Case.Aux[Layer.Aux[Input, Array2D#Batch],
                                                                         Layer.Aux[Input, Array2D#Batch],
                                                                         Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods.*.at { (leftLayer, rightLayer) =>
      MultiplyArray2D(leftLayer, rightLayer)
    }
  }

  implicit def `Array2D*Double`[Input <: Batch]: PolyMethods.*.Case.Aux[Layer.Aux[Input, Array2D#Batch],
                                                                        Layer.Aux[Input, BpDouble#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods.*.at { (leftLayer, rightLayer) =>
      MultiplyDouble(leftLayer, rightLayer)
    }
  }

  implicit def `Double*Array2D`[Input <: Batch]: PolyMethods.*.Case.Aux[Layer.Aux[Input, BpDouble#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods.*.at { (leftLayer, rightLayer) =>
      MultiplyDouble(rightLayer, leftLayer)
    }
  }

  implicit def `Array2D-Array2D`[Input <: Batch]: PolyMethods.-.Case.Aux[Layer.Aux[Input, Array2D#Batch],
                                                                         Layer.Aux[Input, Array2D#Batch],
                                                                         Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods.-.at { (leftLayer, rightLayer) =>
      PlusArray2D(leftLayer, Negative(rightLayer))
    }
  }

  implicit def `Double-Array2D`[Input <: Batch]: PolyMethods.-.Case.Aux[Layer.Aux[Input, BpDouble#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods.-.at { (leftLayer, rightLayer) =>
      PlusDouble(Negative(rightLayer), leftLayer)
    }
  }

  implicit def `Array2D-Double`[Input <: Batch]: PolyMethods.-.Case.Aux[Layer.Aux[Input, Array2D#Batch],
                                                                        Layer.Aux[Input, BpDouble#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods.-.at { (leftLayer, rightLayer) =>
      PlusDouble(leftLayer, double.layers.Negative(rightLayer))
    }
  }

  implicit def `Array2D+Array2D`[Input <: Batch]: PolyMethods.+.Case.Aux[Layer.Aux[Input, Array2D#Batch],
                                                                         Layer.Aux[Input, Array2D#Batch],
                                                                         Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods.+.at { (leftLayer, rightLayer) =>
      PlusArray2D(leftLayer, rightLayer)
    }
  }

  implicit def `Array2D+Double`[Input <: Batch]: PolyMethods.+.Case.Aux[Layer.Aux[Input, Array2D#Batch],
                                                                        Layer.Aux[Input, BpDouble#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods.+.at { (leftLayer, rightLayer) =>
      PlusDouble(leftLayer, rightLayer)
    }
  }

  implicit def `Double+Array2D`[Input <: Batch]: PolyMethods.+.Case.Aux[Layer.Aux[Input, BpDouble#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods.+.at { (leftLayer, rightLayer) =>
      PlusDouble(rightLayer, leftLayer)
    }
  }

  implicit def `exp(Array2D)`[Input <: Batch]
    : exp.Case.Aux[Layer.Aux[Input, Array2D#Batch], Layer.Aux[Input, Array2D#Batch]] = {
    exp.at(Exp(_))
  }

  final class Array2DOps[Input <: Batch](differentiable: Layer.Aux[Input, Array2D#Batch]) {

    def dot(right: Layer.Aux[Input, Array2D#Batch]): Layer.Aux[Input, Array2D#Batch] = {
      Dot(differentiable, right)
    }

    def unary_- : Layer.Aux[Input, Array2D#Batch] = {
      Negative(differentiable)
    }

    def toSeq: Layer.Aux[
      Input,
      Batch.Aux[Seq[Seq[Eval[Double]]], (Int, (Int, Eval[Double]))]] = {
      ToSeq(differentiable)
    }

  }

  implicit def toArray2DOps[From, Input <: Batch](from: From)(
      implicit toLayer: ToLayer.OfType[From, Input, Array2D]
  ): Array2DOps[Input] = {
    new Array2DOps(toLayer(from))
  }

  // TODO: Support Array for better performance.
  final class ToArray2DOps[Input <: Batch](
      layerVector: Seq[Seq[Layer.Aux[Input, Batch.Aux[Eval[Double], Eval[Double]]]]]) {
    def toArray2D: Layer.Aux[Input, Array2D#Batch] = ToArray2D(layerVector)
  }

  implicit def toToArray2DOps[Element, Input <: Batch](layerVector: Seq[Seq[Element]])(
      implicit toLayer: ToLayer.OfType[Element, Input, BpDouble]): ToArray2DOps[Input] = {
    new ToArray2DOps(layerVector.view.map(_.view.map(toLayer(_))))
  }

  implicit final class INDArrayOps(nativeDouble: INDArray) {
    def toWeight[InputData, InputDelta](
                                         implicit inputType: BackPropagationType[InputData, InputDelta],
                                         optimizer: Optimizer): Layer.Aux[Batch.Aux[InputData, InputDelta], Array2D#Batch] = {
      Weight(nativeDouble)
    }
  }

}
