package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.NeuralNetwork.Aux
import com.thoughtworks.deepLearning.array2D.ast.{ToArray2D, MaxDouble, ToSeq}
import com.thoughtworks.deepLearning.double.utilities.Double
import com.thoughtworks.deepLearning.seq2D.utilities.Seq2D
import shapeless.PolyDefns._
import shapeless.{Lazy, Poly2}

import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object array2D {

  /** @template */
  type Array2D = utilities.Array2D

  implicit def `max(Array2D,Double)`[Left, Right, Input <: Batch]
    : max.Case.Aux[NeuralNetwork.Aux[Input, Array2D#Batch],
                   NeuralNetwork.Aux[Input, Double#Batch],
                   NeuralNetwork.Aux[Input, Array2D#Batch]] =
    max.at { MaxDouble(_, _) }

  final class Array2DOps[Input <: Batch](differentiable: NeuralNetwork.Aux[Input, Array2D#Batch]) {
//
//    def dot(right: NeuralNetwork.Aux[Input, Array2D#Batch])
//      : NeuralNetwork.Aux[Input, Array2D#Batch] = {
//      Dot(boolean, right)
//    }
//    def +(right: NeuralNetwork.Aux[Input, Array2D#Batch])
//      : NeuralNetwork.Aux[Input, Array2D#Batch] = {
//      AddArray2D(boolean, right)
//    }
//
//    def unary_- : NeuralNetwork.Aux[Input, Array2D#Batch] = {
//      Negative(boolean)
//    }
//
    def toSeq: NeuralNetwork.Aux[Input, Seq2D#Batch] = {
      ToSeq(differentiable)
    }

  }

  implicit def toArray2DOps[From, Input <: Batch](from: From)(
      implicit toNeuralNetwork: ToNeuralNetwork.OfType[From, Input, Array2D]
  ): Array2DOps[Input] = {
    new Array2DOps(toNeuralNetwork(from))
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
//      override type Out = NeuralNetwork.Aux[Input, Array2D#Batch]
//      override def apply(leftOperand: NeuralNetwork.Aux[Input, Batch.Aux[Eval[INDArray], Eval[INDArray]]],
//                         rightOperand: NeuralNetwork.Aux[Input, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]) = {
//        MaxDouble(leftOperand, rightOperand)
//      }
//    }
////  implicit def array2DMaxDouble[Input <: Batch, Left, Right](
////      implicit leftView: ToNeuralNetwork[Left, Input, Eval[INDArray], Eval[INDArray]],
////      rightView: ToNeuralNetwork[Right, Input, Eval[scala.Double], Eval[scala.Double]]) =
////    max.at[Left, Right].apply[NeuralNetwork.Aux[Input, Array2D#Batch]] { (left, right) =>
////      MaxDouble(leftView(left), rightView(right))
////    }
//
//  implicit final class INDArrayOps(ndarray: INDArray) {
//    def toWeight[Input <: Batch: Identity](
//        implicit learningRate: LearningRate): NeuralNetwork.Aux[Input, Array2D#Batch] =
//      Weight(ndarray)
//    def toLiteral[Input <: Batch: Identity] = ndarrayLiteral(ndarray)
//    def toBatch = ndarrayBatch(ndarray)
//  }
//
//  implicit def nativeArrayToINDArrayOps(nativeArray: Array[Array[scala.Double]]): INDArrayOps =
//    new INDArrayOps(nativeArray.toNDArray)
//
//  implicit def ndarrayLiteral[Input <: Batch: Identity](
//      ndarray: INDArray): NeuralNetwork.Aux[Input, Array2D#Batch] =
//    Literal(Eval.now(ndarray))
//
//  implicit def ndarrayBatch(ndarray: INDArray): Array2D#Batch =
//    Literal(Eval.now(ndarray))
//
//  implicit def nativeArrayLiteral[Input <: Batch: Identity](
//      nativeArray: Array[Array[scala.Double]]): NeuralNetwork.Aux[Input, Array2D#Batch] =
//    ndarrayLiteral(nativeArray.toNDArray)
//
//  implicit def nativeArrayBatch(nativeArray: Array[Array[scala.Double]]): Array2D#Batch =
//    ndarrayBatch(nativeArray.toNDArray)
//
//  // TODO: Support scala.Array for better performance.

  final class ToArray2DOps[Input <: Batch](
      astVector: Seq[Seq[NeuralNetwork.Aux[Input, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]]]) {
    def toArray2D: NeuralNetwork.Aux[Input, Array2D#Batch] = ToArray2D(astVector)
  }

  implicit def toToArray2DOps[Element, Input <: Batch](astVector: Seq[Seq[Element]])(
      implicit toNeuralNetwork: ToNeuralNetwork.OfType[Element, Input, Double]): ToArray2DOps[Input] = {
    new ToArray2DOps(astVector.view.map(_.view.map(toNeuralNetwork(_))))
  }

}
