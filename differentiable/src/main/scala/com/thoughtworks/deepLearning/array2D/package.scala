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

  implicit final class Array2DOps[Input <: Differentiable](differentiable: Ast[Input, Array2D#Widen]) {

    def dot[RightInput <: Input](right: Ast[RightInput, Array2D#Widen]): Ast[RightInput, Array2D#Widen] = {
      Dot(differentiable, right)
    }
    def +[RightInput <: Input](right: Ast[RightInput, Array2D#Widen]): Ast[RightInput, Array2D#Widen] = {
      AddArray2D(differentiable, right)
    }

    def unary_- : Ast[Input, Array2D#Widen] = {
      Negative(differentiable)
    }

    def toSeq: Ast[Input, Seq2D#Widen] = {
      ToSeq(differentiable)
    }

    def max[RightInput <: Input](rightAst: Ast[RightInput, Double#Widen]): Ast[RightInput, Array2D#Widen] = {
      MaxDouble(differentiable, rightAst)
    }
  }

  implicit final class INDArrayOps(ndarray: INDArray) {
    def toWeight[Input <: Differentiable: Identity](implicit learningRate: LearningRate): Ast[Input, Array2D#Widen] =
      Weight(ndarray)
    def toLiteral[Input <: Differentiable: Identity] = ndarrayLiteral(ndarray)
    def toBatch = ndarrayBatch(ndarray)
  }

  implicit def nativeArrayToINDArrayOps(nativeArray: Array[Array[scala.Double]]): INDArrayOps =
    new INDArrayOps(nativeArray.toNDArray)

  implicit def ndarrayLiteral[Input <: Differentiable: Identity](ndarray: INDArray): Ast[Input, Array2D#Widen] =
    Literal(Eval.now(ndarray))

  implicit def ndarrayBatch(ndarray: INDArray): Array2D#Widen =
    Literal(Eval.now(ndarray))

  implicit def nativeArrayLiteral[Input <: Differentiable: Identity](
      nativeArray: Array[Array[scala.Double]]): Ast[Input, Array2D#Widen] =
    ndarrayLiteral(nativeArray.toNDArray)

  implicit def nativeArrayBatch(nativeArray: Array[Array[scala.Double]]): Array2D#Widen =
    ndarrayBatch(nativeArray.toNDArray)

  // TODO: Support scala.Array for better performance.
  implicit final class AstVectorOps[Input <: Differentiable](
      astVector: Vector[Vector[Ast[Input, Batch[Eval[scala.Double], Eval[scala.Double]]]]]) {
    def toArray2D = FromAstVector(astVector)
  }

}
