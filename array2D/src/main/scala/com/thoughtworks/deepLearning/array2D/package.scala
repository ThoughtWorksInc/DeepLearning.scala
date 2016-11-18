package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.any._
import com.thoughtworks.deepLearning.array2D.layers._
import com.thoughtworks.deepLearning.double.utilities.Double
import com.thoughtworks.deepLearning.seq2D.utilities.Seq2D
import org.nd4j.linalg.api.ndarray.INDArray

import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object array2D {

  /** @template */
  type Array2D = utilities.Array2D

  implicit def `max(Array2D,Double)`[Left, Right, Input <: Batch]
    : max.Case.Aux[Layer.Aux[Input, Array2D#Batch], Layer.Aux[Input, Double#Batch], Layer.Aux[Input, Array2D#Batch]] =
    max.at { MaxDouble(_, _) }

  implicit def `Array2D/Array2D`[Input <: Batch]: PolyMethods./.Case.Aux[Layer.Aux[Input, Array2D#Batch],
                                                                         Layer.Aux[Input, Array2D#Batch],
                                                                         Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods./.at { (leftLayer, rightLayer) =>
      MultiplyArray2D(leftLayer, Reciprocal(rightLayer))
    }
  }
  implicit def `Double/Array2D`[Input <: Batch]: PolyMethods./.Case.Aux[Layer.Aux[Input, Double#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods./.at { (leftLayer, rightLayer) =>
      MultiplyDouble(Reciprocal(rightLayer), leftLayer)
    }
  }
  implicit def `Array2D/Double`[Input <: Batch]: PolyMethods./.Case.Aux[Layer.Aux[Input, Array2D#Batch],
                                                                        Layer.Aux[Input, Double#Batch],
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
                                                                        Layer.Aux[Input, Double#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods.*.at { (leftLayer, rightLayer) =>
      MultiplyDouble(leftLayer, rightLayer)
    }
  }
  implicit def `Double*Array2D`[Input <: Batch]: PolyMethods.*.Case.Aux[Layer.Aux[Input, Double#Batch],
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
  implicit def `Double-Array2D`[Input <: Batch]: PolyMethods.-.Case.Aux[Layer.Aux[Input, Double#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods.-.at { (leftLayer, rightLayer) =>
      PlusDouble(Negative(rightLayer), leftLayer)
    }
  }
  implicit def `Array2D-Double`[Input <: Batch]: PolyMethods.-.Case.Aux[Layer.Aux[Input, Array2D#Batch],
                                                                        Layer.Aux[Input, Double#Batch],
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
                                                                        Layer.Aux[Input, Double#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods.+.at { (leftLayer, rightLayer) =>
      PlusDouble(leftLayer, rightLayer)
    }
  }
  implicit def `Double+Array2D`[Input <: Batch]: PolyMethods.+.Case.Aux[Layer.Aux[Input, Double#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch],
                                                                        Layer.Aux[Input, Array2D#Batch]] = {
    PolyMethods.+.at { (leftLayer, rightLayer) =>
      PlusDouble(rightLayer, leftLayer)
    }
  }

  final class Array2DOps[Input <: Batch](differentiable: Layer.Aux[Input, Array2D#Batch]) {

    def dot(right: Layer.Aux[Input, Array2D#Batch]): Layer.Aux[Input, Array2D#Batch] = {
      Dot(differentiable, right)
    }

    def unary_- : Layer.Aux[Input, Array2D#Batch] = {
      Negative(differentiable)
    }

    def toSeq: Layer.Aux[Input, Seq2D#Batch] = {
      ToSeq(differentiable)
    }

  }

  implicit def toArray2DOps[From, Input <: Batch](from: From)(
      implicit toLayer: ToLayer.OfType[From, Input, Array2D]
  ): Array2DOps[Input] = {
    new Array2DOps(toLayer(from))
  }

  ////
////  private[array2D] trait Case2Double { this: Poly2#Case =>
////    override type LeftOperandData = Eval[INDArray]
////    override type LeftOperandDelta = Eval[INDArray]
////    override type RightOperandData = Eval[scala.Double]
////    override type RightOperandDelta = Eval[scala.Double]
////    override type OutputData = Eval[INDArray]
////    override type OutputDelta = Eval[INDArray]
////  }
//
//  implicit def maxArray2DDouble[Input <: Batch] =
//    new max.Case[Input, Eval[INDArray], Eval[INDArray], Eval[scala.Double], Eval[scala.Double]] {
//      override type Out = Layer.Aux[Input, Array2D#Batch]
//      override def apply(leftOperand: Layer.Aux[Input, Batch.Aux[Eval[INDArray], Eval[INDArray]]],
//                         rightOperand: Layer.Aux[Input, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]) = {
//        MaxDouble(leftOperand, rightOperand)
//      }
//    }
////  implicit def array2DMaxDouble[Input <: Batch, Left, Right](
////      implicit leftView: ToLayer[Left, Input, Eval[INDArray], Eval[INDArray]],
////      rightView: ToLayer[Right, Input, Eval[scala.Double], Eval[scala.Double]]) =
////    max.at[Left, Right].apply[Layer.Aux[Input, Array2D#Batch]] { (left, right) =>
////      MaxDouble(leftView(left), rightView(right))
////    }
//
//  implicit final class INDArrayOps(ndarray: INDArray) {
//    def toWeight[Input <: Batch: Identity](
//        implicit learningRate: LearningRate): Layer.Aux[Input, Array2D#Batch] =
//      Weight(ndarray)
//    def toLiteral[Input <: Batch: Identity] = ndarrayLiteral(ndarray)
//    def toBatchId = ndarrayBatch(ndarray)
//  }
//
//  implicit def nativeArrayToINDArrayOps(nativeArray: Array[Array[scala.Double]]): INDArrayOps =
//    new INDArrayOps(nativeArray.toNDArray)
//
//  implicit def ndarrayLiteral[Input <: Batch: Identity](
//      ndarray: INDArray): Layer.Aux[Input, Array2D#Batch] =
//    Literal(Eval.now(ndarray))
//
//  implicit def ndarrayBatch(ndarray: INDArray): Array2D#Batch =
//    Literal(Eval.now(ndarray))
//
//  implicit def nativeArrayLiteral[Input <: Batch: Identity](
//      nativeArray: Array[Array[scala.Double]]): Layer.Aux[Input, Array2D#Batch] =
//    ndarrayLiteral(nativeArray.toNDArray)
//
//  implicit def nativeArrayBatch(nativeArray: Array[Array[scala.Double]]): Array2D#Batch =
//    ndarrayBatch(nativeArray.toNDArray)
//
//  // TODO: Support scala.Array for better performance.

  final class ToArray2DOps[Input <: Batch](
      layerVector: Seq[Seq[Layer.Aux[Input, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]]]) {
    def toArray2D: Layer.Aux[Input, Array2D#Batch] = ToArray2D(layerVector)
  }

  implicit def toToArray2DOps[Element, Input <: Batch](layerVector: Seq[Seq[Element]])(
      implicit toLayer: ToLayer.OfType[Element, Input, Double]): ToArray2DOps[Input] = {
    new ToArray2DOps(layerVector.view.map(_.view.map(toLayer(_))))
  }

  implicit final class INDArrayOps(nativeDouble: INDArray) {
    def toWeight[InputData, InputDelta](
        implicit inputType: Type[InputData, InputDelta],
        optimizer: Optimizer): Layer.Aux[Batch.Aux[InputData, InputDelta], Array2D#Batch] = {
      Weight(nativeDouble)
    }
  }

}
