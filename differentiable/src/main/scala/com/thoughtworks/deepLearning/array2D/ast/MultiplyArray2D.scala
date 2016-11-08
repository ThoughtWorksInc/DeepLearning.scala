package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.core.DifferentiableFunction._
import com.thoughtworks.deepLearning.core.Differentiable._
import org.nd4j.linalg.api.ndarray.INDArray
import com.thoughtworks.deepLearning.array2D.utilities._
import org.nd4s.Implicits._
import SumAs._
import com.thoughtworks.deepLearning.core.{Differentiable, DifferentiableFunction}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class MultiplyArray2D[Input0 <: Differentiable](
                                                            leftOperand: DifferentiableFunction.Ast[Input0, Array2D#Batch],
                                                            rightOperand: DifferentiableFunction.Ast[Input0, Array2D#Batch]
) extends DifferentiableFunction
    with Cached {

  protected final class SharedBatch private[deepLearning](override val input: Input0,
                                                          upstream1: Array2D#Batch,
                                                          upstream2: Array2D#Batch)
      extends Array2DSemigroupBatch
      with SemigroupBatch {
    val value = {
      upstream1.value
        .map2(upstream2.value) { (aValue, bValue) =>
          val Array(aRows, aColumns) = aValue.shape()
          val Array(bRows, bColumns) = bValue.shape()
          val newShape =
            Array(math.max(aRows, bRows), math.max(aColumns, bColumns))
          aValue.broadcast(newShape: _*) * bValue.broadcast(newShape: _*)
        }
        .memoize
    }

    type Input >: Input0

    override protected def closeUpstreams(): Unit = {
      upstream1.close()
      upstream2.close()
    }

    override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
      val a = upstream1.value
      val b = upstream2.value
      upstream1.backward(
        Applicative[Eval]
          .map3(outputDelta, a, b) { (outputDeltaValue, aData, bData) =>
            sumAs(bData.broadcast(outputDeltaValue.shape(): _*) * outputDeltaValue, aData.shape())
          }
          .memoize)
      upstream2.backward(
        Applicative[Eval]
          .map3(outputDelta, a, b) { (outputDeltaValue, aData, bData) =>
            sumAs(aData.broadcast(outputDeltaValue.shape(): _*) * outputDeltaValue, bData.shape())
          }
          .memoize)
    }
  }

  type Input = Input0

  override protected def rawForward(input: Input): SharedBatch = {
    new SharedBatch(input, leftOperand.forward(input), rightOperand.forward(input))
  }
}
