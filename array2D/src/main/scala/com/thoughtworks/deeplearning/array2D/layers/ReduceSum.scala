package com.thoughtworks.deeplearning
package array2D.layers

import cats._
import cats.implicits._
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.Batch._
import com.thoughtworks.deeplearning.array2D.utilities._
import com.thoughtworks.deeplearning.double.utilities.DoubleMonoidBatch
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class ReduceSum[Input0 <: Batch](operand: Layer.Aux[Input0, Array2D#Batch]) extends BufferedLayer.Unary {
  type BufferedBatch = DoubleMonoidBatch with MonoidBatch with UnaryBatch

  type Input = Input0

  override protected def rawForward(input0: Input): BufferedBatch = {
    new {
      override val input = input0
    } with DoubleMonoidBatch with MonoidBatch with UnaryBatch {

      val value = upstream.value.map(_.sumT).memoize

      override protected def rawBackward(outputDelta: Eval[Double]): Unit = {
        upstream.backward(
          outputDelta
            .map2(upstream.value) { (outputDeltaValue, aValue) =>
              Nd4j.valueArrayOf(aValue.shape(), outputDeltaValue)
            }
            .memoize)
      }
    }
  }
}
