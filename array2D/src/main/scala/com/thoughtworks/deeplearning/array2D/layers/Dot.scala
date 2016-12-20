package com.thoughtworks.deeplearning
package array2D.layers

import cats._
import cats.implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import com.thoughtworks.deeplearning.array2D.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Dot[Input0 <: Batch](
    operand1: Layer.Aux[Input0, Array2D#Batch],
    operand2: Layer.Aux[Input0, Array2D#Batch]
) extends BufferedLayer.Binary {

  type BufferedBatch = Array2DSemigroupBatch with SemigroupBatch with BinaryBatch

  type Input = Input0

  override protected def rawForward(input0: Input): BufferedBatch = {
    new {
      override final val input = input0
    } with Array2DSemigroupBatch with SemigroupBatch with BinaryBatch {

      override val value = upstream1.value.map2(upstream2.value)(_ dot _).memoize

      override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
        val b = upstream2.value
        upstream1.backward(
          outputDelta
            .map2(b) {
              _ dot _.T
            }
            .memoize)
        val a = upstream1.value
        upstream2.backward(
          outputDelta
            .flatMap[INDArray] { outputDeltaValue =>
              a.map { aData =>
                aData.T.dot(outputDeltaValue)
              }
            }
            .memoize)
      }
    }

  }
}
