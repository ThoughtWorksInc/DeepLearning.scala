package com.thoughtworks.deeplearning
package array2D.layers

import cats._
import cats.implicits._
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.Batch._
import org.nd4j.linalg.api.ndarray.INDArray
import com.thoughtworks.deeplearning.array2D.utilities._
import org.nd4s.Implicits._
import SumAs._
import org.nd4j.linalg.ops.transforms.Transforms

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class MultiplyArray2D[Input0 <: Batch](
    operand1: Layer.Aux[Input0, Array2D#Batch],
    operand2: Layer.Aux[Input0, Array2D#Batch]
) extends BufferedLayer.Binary {

  type BufferedBatch = Array2DSemigroupBatch with SemigroupBatch with BinaryBatch

  type Input = Input0

  override protected def rawForward(input0: Input): BufferedBatch = {
    new {
      override final val input = input0
    } with Array2DSemigroupBatch with SemigroupBatch with BinaryBatch {

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
  }
}
