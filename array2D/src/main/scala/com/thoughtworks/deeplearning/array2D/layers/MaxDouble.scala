package com.thoughtworks.deeplearning
package array2D.layers

import cats._
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.Batch._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import cats.implicits._
import com.thoughtworks.deeplearning.BufferedLayer
import com.thoughtworks.deeplearning.array2D.utilities._
import com.thoughtworks.deeplearning.double.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class MaxDouble[Input0 <: Batch](
    leftOperand: Layer.Aux[Input0, Array2D#Batch],
    rightOperand: Layer.Aux[Input0, Double#Batch]
) extends Layer
    with BufferedLayer {

  protected final class BufferedBatch private[deeplearning] (override val input: BatchId.Aux[Input0],
                                                             upstream1: Array2D#Batch,
                                                             upstream2: Double#Batch)
      extends Array2DSemigroupBatch
      with SemigroupBatch {
    val value = upstream1.value.map2(upstream2.value)(Transforms.max).memoize

    override protected def closeUpstreams(): Unit = {
      upstream1.close()
      upstream2.close()
    }

    override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
      val a = upstream1.value
      val b = upstream2.value
      upstream1.backward(
        Applicative[Eval].map3(outputDelta, a, b) { (outputDeltaValue, aValue, bValue) =>
          (aValue gt bValue) * outputDeltaValue
        }
      )
      upstream2.backward(
        Applicative[Eval].map3(outputDelta, a, b) { (outputDeltaValue, aValue, bValue) =>
          ((aValue lt bValue) * outputDeltaValue).sumT
        }
      )
    }
  }

  type Input = Input0

  override protected def rawForward(input: BatchId.Aux[Input]): BufferedBatch = {
    new BufferedBatch(input, leftOperand.forward(input).open(), rightOperand.forward(input).open())
  }
}
