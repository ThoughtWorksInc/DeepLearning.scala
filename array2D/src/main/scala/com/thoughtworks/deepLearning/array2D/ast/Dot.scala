package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.NeuralNetwork._
import com.thoughtworks.deepLearning.Batch._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning.array2D.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Dot[Input0 <: Batch](
    leftOperand: NeuralNetwork.Aux[Input0, Array2D#Batch],
    rightOperand: NeuralNetwork.Aux[Input0, Array2D#Batch]
) extends NeuralNetwork
    with BufferedNetwork {

  protected final class BufferedBatch private[deepLearning] (override val input: BatchId.Aux[Input0],
                                                             upstream1: Array2D#Batch,
                                                             upstream2: Array2D#Batch)
      extends Array2DSemigroupBatch
      with SemigroupBatch {
    override val value = upstream1.value.map2(upstream2.value)(_ dot _).memoize

    override protected def closeUpstreams(): Unit = {
      upstream1.close()
      upstream2.close()
    }

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

  type Input = Input0

  override protected def rawForward(input: BatchId.Aux[Input]): BufferedBatch = {
    new BufferedBatch(input, leftOperand.forward(input).open(), rightOperand.forward(input).open())
  }
}
