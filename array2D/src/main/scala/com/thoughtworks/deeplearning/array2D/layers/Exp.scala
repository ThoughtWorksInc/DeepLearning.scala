package com.thoughtworks.deeplearning
package array2D.layers

import cats._
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.Batch._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import cats.implicits._
import com.thoughtworks.deeplearning.BufferedLayer
import com.thoughtworks.deeplearning.array2D.utilities._
import org.nd4s.Implicits._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Exp[Input0 <: Batch](operand: Layer.Aux[Input0, Array2D#Batch]) extends BufferedLayer.Unary {
  type BufferedBatch = Array2DSemigroupBatch with SemigroupBatch with UnaryBatch

  type Input = Input0

  override protected def rawForward(input0: Input): BufferedBatch = {
    new {
      override val input = input0
    } with Array2DSemigroupBatch with SemigroupBatch with UnaryBatch {
      val value = upstream.value.map(Transforms.exp).memoize

      override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
        upstream.backward(value.map2(outputDelta)(_ * _).memoize)
      }
    }
  }
}
