package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.DifferentiableFunction.Ast
import com.thoughtworks.deepLearning.array2D.ast.MaxDouble
import com.thoughtworks.deepLearning.double.utilities.Double

//
//import cats.Eval
//import com.thoughtworks.deepLearning.DifferentiableFunction.{Ast, ToAst}
//import com.thoughtworks.deepLearning.Differentiable.ConcreteBatch
//import com.thoughtworks.deepLearning.Poly.Poly2
//import com.thoughtworks.deepLearning.any.ast.{Identity, Literal}
//import com.thoughtworks.deepLearning.array2D.ast._
//import com.thoughtworks.deepLearning.seq2D.utilities.Seq2D
//import com.thoughtworks.deepLearning.double.utilities.Double
//import org.nd4j.linalg.api.ndarray.INDArray
//import org.nd4s.Implicits._
//
//import scala.language.implicitConversions
//
/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object array2D {

  /** @template */
  type Array2D = utilities.Array2D

  implicit def maxArray2DDouble[Left, Right, Input <: Differentiable](
      implicit leftToAst: ToAst.OfType[Left, Input, Array2D],
      rightToAst: ToAst.OfType[Right, Input, Double]): max.Case.Aux[Left, Right, Ast[Input, Array2D#Batch]] =
    max.at { (left, right) =>
      val leftAst = leftToAst(left)
      val rightAst = rightToAst(right)
      MaxDouble(leftAst, rightAst)
//      If[Input, Double#Batch](LessThan[Input](leftAst, rightAst), rightAst, leftAst)
    }
//
//  implicit final class Array2DOps[Input <: Differentiable](
//      differentiable: DifferentiableFunction.Ast[Input, Array2D#ConcreteBatch]) {
//
//    def dot(right: DifferentiableFunction.Ast[Input, Array2D#ConcreteBatch])
//      : DifferentiableFunction.Ast[Input, Array2D#ConcreteBatch] = {
//      Dot(differentiable, right)
//    }
//    def +(right: DifferentiableFunction.Ast[Input, Array2D#ConcreteBatch])
//      : DifferentiableFunction.Ast[Input, Array2D#ConcreteBatch] = {
//      AddArray2D(differentiable, right)
//    }
//
//    def unary_- : DifferentiableFunction.Ast[Input, Array2D#ConcreteBatch] = {
//      Negative(differentiable)
//    }
//
//    def toSeq: DifferentiableFunction.Ast[Input, Seq2D#ConcreteBatch] = {
//      ToSeq(differentiable)
//    }
//
//  }
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
//  implicit def maxArray2DDouble[Input <: Differentiable] =
//    new max.Case[Input, Eval[INDArray], Eval[INDArray], Eval[scala.Double], Eval[scala.Double]] {
//      override type Out = Ast[Input, Array2D#ConcreteBatch]
//      override def apply(leftOperand: Ast[Input, ConcreteBatch[Eval[INDArray], Eval[INDArray]]],
//                         rightOperand: Ast[Input, ConcreteBatch[Eval[scala.Double], Eval[scala.Double]]]) = {
//        MaxDouble(leftOperand, rightOperand)
//      }
//    }
////  implicit def array2DMaxDouble[Input <: Differentiable, Left, Right](
////      implicit leftView: ToAst[Left, Input, Eval[INDArray], Eval[INDArray]],
////      rightView: ToAst[Right, Input, Eval[scala.Double], Eval[scala.Double]]) =
////    max.at[Left, Right].apply[DifferentiableFunction.Ast[Input, Array2D#ConcreteBatch]] { (left, right) =>
////      MaxDouble(leftView(left), rightView(right))
////    }
//
//  implicit final class INDArrayOps(ndarray: INDArray) {
//    def toWeight[Input <: Differentiable: Identity](
//        implicit learningRate: LearningRate): DifferentiableFunction.Ast[Input, Array2D#ConcreteBatch] =
//      Weight(ndarray)
//    def toLiteral[Input <: Differentiable: Identity] = ndarrayLiteral(ndarray)
//    def toBatch = ndarrayBatch(ndarray)
//  }
//
//  implicit def nativeArrayToINDArrayOps(nativeArray: Array[Array[scala.Double]]): INDArrayOps =
//    new INDArrayOps(nativeArray.toNDArray)
//
//  implicit def ndarrayLiteral[Input <: Differentiable: Identity](
//      ndarray: INDArray): DifferentiableFunction.Ast[Input, Array2D#ConcreteBatch] =
//    Literal(Eval.now(ndarray))
//
//  implicit def ndarrayBatch(ndarray: INDArray): Array2D#ConcreteBatch =
//    Literal(Eval.now(ndarray))
//
//  implicit def nativeArrayLiteral[Input <: Differentiable: Identity](
//      nativeArray: Array[Array[scala.Double]]): DifferentiableFunction.Ast[Input, Array2D#ConcreteBatch] =
//    ndarrayLiteral(nativeArray.toNDArray)
//
//  implicit def nativeArrayBatch(nativeArray: Array[Array[scala.Double]]): Array2D#ConcreteBatch =
//    ndarrayBatch(nativeArray.toNDArray)
//
//  // TODO: Support scala.Array for better performance.
//  implicit final class AstVectorOps[Input <: Differentiable](
//      astVector: Vector[
//        Vector[DifferentiableFunction.Ast[Input, Differentiable.ConcreteBatch[Eval[scala.Double], Eval[scala.Double]]]]]) {
//    def toArray2D = FromAstVector(astVector)
//  }

}
