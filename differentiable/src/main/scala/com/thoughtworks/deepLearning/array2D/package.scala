package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.any.ast.Literal
import com.thoughtworks.deepLearning.array2D.ast.{Dot, Negative, ToArray2D, Weight}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object array2D {

  type Array2D = {
    type Data = Eval[INDArray]
    type Delta = Eval[INDArray]
  }

  implicit final class Array2DOps[Input <: Batch](
      differentiable: Ast.Aux[Input, Batch.Aux[Eval[INDArray], Eval[INDArray]]]) {

    def dot[RightInput <: Input](right: Ast.Aux[RightInput, Batch.Aux[Eval[INDArray], Eval[INDArray]]]) = {
      Dot(differentiable, right)
    }

    def unary_- = {
      Negative(differentiable)
    }

  }

  implicit final class NativeArrayOps(nativeArray: Array[Array[scala.Double]]) {
    def toWeight(implicit learningRate: LearningRate) = Weight(nativeArray.toNDArray)
    def toLiteral = array2DLiteral(nativeArray)
  }

  def randn(numberOfRows: Int, numberOfColumns: Int)(implicit learningRate: LearningRate) = {
    Weight(Nd4j.randn(numberOfRows, numberOfColumns))
  }

  def zeros(numberOfRows: Int, numberOfColumns: Int)(implicit learningRate: LearningRate) = {
    Weight(Nd4j.zeros(numberOfRows, numberOfColumns))
  }

  implicit def array2DLiteral(nativeArray: Array[Array[scala.Double]]) = Literal(Eval.now(nativeArray.toNDArray))

  def make[Input <: Batch](operands: Vector[Vector[Ast.Aux[Input, Batch.Aux[Eval[Double], Eval[Double]]]]]) = {
    ToArray2D(operands)
  }

}
