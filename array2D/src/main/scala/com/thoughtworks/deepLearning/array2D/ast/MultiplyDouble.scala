package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.NeuralNetwork._
import com.thoughtworks.deepLearning.Batch._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning.array2D.utilities._
import com.thoughtworks.deepLearning.double.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class MultiplyDouble[Input0 <: Batch](
                                                           leftOperand: NeuralNetwork.Aux[Input0, Array2D#Batch],
                                                           rightOperand: NeuralNetwork.Aux[Input0, Double#Batch]
) extends NeuralNetwork
    with BufferedNetwork {

  protected final class BufferedBatch private[deepLearning](override val input: BatchId.Aux[Input0],
                                                            upstream1: Array2D#Batch,
                                                            upstream2: Double#Batch)
      extends Array2DSemigroupBatch
      with SemigroupBatch {
    val value = upstream1.value.map2(upstream2.value)(_ * _).memoize



    override protected def closeUpstreams(): Unit = {
      upstream1.close()
      upstream2.close()
    }

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

  type Input = Input0

  override protected def rawForward(input: BatchId.Aux[Input]): BufferedBatch = {
    new BufferedBatch(input, leftOperand.forward(input).open(), rightOperand.forward(input).open())
  }
}
