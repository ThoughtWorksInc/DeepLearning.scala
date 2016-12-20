package com.thoughtworks.deeplearning
package array2D.layers

import cats._
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.Layer.Batch._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import cats.implicits._
import com.thoughtworks.deeplearning.BufferedLayer
import com.thoughtworks.deeplearning.double.utilities.BpDouble
import com.thoughtworks.deeplearning.array2D.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class PlusDouble[Input0 <: Batch](
    operand1: Layer.Aux[Input0, Array2D#Batch],
    operand2: Layer.Aux[Input0, BpDouble#Batch]
) extends BufferedLayer.Binary {

  type BufferedBatch = Array2DSemigroupBatch with SemigroupBatch with BinaryBatch

  type Input = Input0

  override protected def rawForward(input0: Input): BufferedBatch = {
    new {
      override final val input = input0
    } with Array2DSemigroupBatch with SemigroupBatch with BinaryBatch {
      val value = upstream1.value.map2(upstream2.value)(_ + _).memoize

      override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
        upstream1.backward(outputDelta)
        upstream2.backward(outputDelta.map(_.sumT))
      }
    }
  }
}
