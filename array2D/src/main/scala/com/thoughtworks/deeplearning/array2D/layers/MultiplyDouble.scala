package com.thoughtworks.deeplearning
package array2D.layers

import cats._
import cats.implicits._
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.Batch._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import com.thoughtworks.deeplearning.array2D.utilities._
import com.thoughtworks.deeplearning.double.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class MultiplyDouble[Input0 <: Batch](
    operand1: Layer.Aux[Input0, Array2D#Batch],
    operand2: Layer.Aux[Input0, Double#Batch]
) extends BufferedLayer.Binary {

  type BufferedBatch = Array2DSemigroupBatch with SemigroupBatch with BinaryBatch

  type Input = Input0

  override protected def rawForward(input0: Input): BufferedBatch = {
    new {
      override final val input = input0
    } with Array2DSemigroupBatch with SemigroupBatch with BinaryBatch {

      val value = upstream1.value.map2(upstream2.value)(_ * _).memoize

      override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
        val a = upstream1.value
        val b = upstream2.value

        val aDelta = outputDelta.map2(b)(_ * _).memoize
        upstream1.backward(aDelta)
        val bDelta = outputDelta
          .map2(a) { (outputDeltaValue, aValue) =>
            (aValue * outputDeltaValue).sumT
          }
          .memoize
        upstream2.backward(bDelta)
      }
    }
  }
}
