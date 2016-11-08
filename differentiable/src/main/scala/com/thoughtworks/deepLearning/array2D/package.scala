package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.DifferentiableFunction.Ast
import com.thoughtworks.deepLearning.Differentiable.Batch
import com.thoughtworks.deepLearning.any.ast.{Identity, Literal}
import com.thoughtworks.deepLearning.array2D.ast._
import com.thoughtworks.deepLearning.seq2D.utilities.Seq2D
import com.thoughtworks.deepLearning.double.utilities.Double
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object array2D {

  /** @template */
  type Array2D = utilities.Array2D

  implicit final class Array2DOps[Input <: Differentiable](differentiable: DifferentiableFunction.Ast[Input, Array2D#Batch]) {

    def dot[RightInput <: Input](right: DifferentiableFunction.Ast[RightInput, Array2D#Batch]): DifferentiableFunction.Ast[RightInput, Array2D#Batch] = {
      Dot(differentiable, right)
    }
    def +[RightInput <: Input](right: DifferentiableFunction.Ast[RightInput, Array2D#Batch]): DifferentiableFunction.Ast[RightInput, Array2D#Batch] = {
      AddArray2D(differentiable, right)
    }

    def unary_- : DifferentiableFunction.Ast[Input, Array2D#Batch] = {
      Negative(differentiable)
    }

    def toSeq: DifferentiableFunction.Ast[Input, Seq2D#Batch] = {
      ToSeq(differentiable)
    }

    def max[RightInput <: Input](rightAst: DifferentiableFunction.Ast[RightInput, Double#Batch]): DifferentiableFunction.Ast[RightInput, Array2D#Batch] = {
      MaxDouble(differentiable, rightAst)
    }
  }

  implicit final class INDArrayOps(ndarray: INDArray) {
    def toWeight[Input <: Differentiable: Identity](implicit learningRate: LearningRate): DifferentiableFunction.Ast[Input, Array2D#Batch] =
      Weight(ndarray)
    def toLiteral[Input <: Differentiable: Identity] = ndarrayLiteral(ndarray)
    def toBatch = ndarrayBatch(ndarray)
  }

  implicit def nativeArrayToINDArrayOps(nativeArray: Array[Array[scala.Double]]): INDArrayOps =
    new INDArrayOps(nativeArray.toNDArray)

  implicit def ndarrayLiteral[Input <: Differentiable: Identity](ndarray: INDArray): DifferentiableFunction.Ast[Input, Array2D#Batch] =
    Literal(Eval.now(ndarray))

  implicit def ndarrayBatch(ndarray: INDArray): Array2D#Batch =
    Literal(Eval.now(ndarray))

  implicit def nativeArrayLiteral[Input <: Differentiable: Identity](
      nativeArray: Array[Array[scala.Double]]): DifferentiableFunction.Ast[Input, Array2D#Batch] =
    ndarrayLiteral(nativeArray.toNDArray)

  implicit def nativeArrayBatch(nativeArray: Array[Array[scala.Double]]): Array2D#Batch =
    ndarrayBatch(nativeArray.toNDArray)

  // TODO: Support scala.Array for better performance.
  implicit final class AstVectorOps[Input <: Differentiable](
      astVector: Vector[Vector[DifferentiableFunction.Ast[Input, Differentiable.Batch[Eval[scala.Double], Eval[scala.Double]]]]]) {
    def toArray2D = FromAstVector(astVector)
  }

}
