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

  type Array2DBatch = Batch.Aux[Eval[INDArray], Eval[INDArray]]

  implicit final class Array2DOps[Input <: Batch](differentiable: Differentiable.Aux[Input, Array2DBatch]) {

    def dot[RightInput <: Input](right: Differentiable.Aux[RightInput, Array2DBatch]) = {
      Dot(differentiable, right)
    }

    def unary_- = {
      Negative(differentiable)
    }

  }

  def randnWeight(numberOfRows: Int, numberOfColumns: Int)(implicit learningRate: LearningRate) = {
    Weight(Nd4j.randn(numberOfRows, numberOfColumns))
  }

  def zerosWeight(numberOfRows: Int, numberOfColumns: Int)(implicit learningRate: LearningRate) = {
    Weight(Nd4j.zeros(numberOfRows, numberOfColumns))
  }

  def weight(nativeArray: Array[Array[scala.Double]])(implicit learningRate: LearningRate) = {
    Weight(nativeArray.toNDArray)
  }

  def literal(nativeArray: Array[Array[scala.Double]]) = {
    Literal(Eval.now(nativeArray.toNDArray))
  }

  implicit def array2DLiteral(nativeArray: Array[Array[scala.Double]]) = literal(nativeArray)

  def make[Input <: Batch](
      operands: Vector[Vector[Differentiable.Aux[Input, Batch.Aux[Eval[Double], Eval[Double]]]]]) = {
    ToArray2D(operands)
  }

}
