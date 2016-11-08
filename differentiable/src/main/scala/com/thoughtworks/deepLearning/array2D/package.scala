package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.Ast.WidenAst
import com.thoughtworks.deepLearning.Batch.WidenBatch
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

  implicit final class Array2DOps[Input <: Batch](differentiable: WidenAst[Input, Array2D#Widen]) {

    def dot[RightInput <: Input](right: WidenAst[RightInput, Array2D#Widen]): WidenAst[RightInput, Array2D#Widen] = {
      Dot(differentiable, right)
    }
    def +[RightInput <: Input](right: WidenAst[RightInput, Array2D#Widen]): WidenAst[RightInput, Array2D#Widen] = {
      AddArray2D(differentiable, right)
    }

    def unary_- : WidenAst[Input, Array2D#Widen] = {
      Negative(differentiable)
    }

    def toSeq: WidenAst[Input, Seq2D#Widen] = {
      ToSeq(differentiable)
    }

    def max[RightInput <: Input](rightAst: WidenAst[RightInput, Double#Widen]): WidenAst[RightInput, Array2D#Widen] = {
      MaxDouble(differentiable, rightAst)
    }
  }

  implicit final class INDArrayOps(ndarray: INDArray) {
    def toWeight[Input <: Batch: Identity](implicit learningRate: LearningRate): WidenAst[Input, Array2D#Widen] =
      Weight(ndarray)
    def toLiteral[Input <: Batch: Identity] = ndarrayLiteral(ndarray)
    def toBatch = ndarrayBatch(ndarray)
  }

  implicit def nativeArrayToINDArrayOps(nativeArray: Array[Array[scala.Double]]): INDArrayOps =
    new INDArrayOps(nativeArray.toNDArray)

  implicit def ndarrayLiteral[Input <: Batch: Identity](ndarray: INDArray): WidenAst[Input, Array2D#Widen] =
    Literal(Eval.now(ndarray))

  implicit def ndarrayBatch(ndarray: INDArray): Array2D#Widen =
    Literal(Eval.now(ndarray))

  implicit def nativeArrayLiteral[Input <: Batch: Identity](
      nativeArray: Array[Array[scala.Double]]): WidenAst[Input, Array2D#Widen] =
    ndarrayLiteral(nativeArray.toNDArray)

  implicit def nativeArrayBatch(nativeArray: Array[Array[scala.Double]]): Array2D#Widen =
    ndarrayBatch(nativeArray.toNDArray)

  // TODO: Support scala.Array for better performance.
  implicit final class AstVectorOps[Input <: Batch](
      astVector: Vector[Vector[WidenAst[Input, WidenBatch[Eval[scala.Double], Eval[scala.Double]]]]]) {
    def toArray2D = FromAstVector(astVector)
  }

}
