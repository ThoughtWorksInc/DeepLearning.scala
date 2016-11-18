package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import com.thoughtworks.deepLearning.NeuralNetwork._
import com.thoughtworks.deepLearning.Batch._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning.array2D.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Negative[Input0 <: Batch](operand: NeuralNetwork.Aux[Input0, Array2D#Batch])
    extends Cached {

  protected final class SharedBatch private[deepLearning](override val input: BatchId.Aux[Input0], upstream: Array2D#Batch)
      extends Array2DSemigroupBatch
      with SemigroupBatch {
    val value = upstream.value.map(-_).memoize

    type Input >: Input0

    override protected def closeUpstreams(): Unit = {
      upstream.close()
    }

    override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
      upstream.backward(outputDelta.map(-_).memoize)
    }
  }

  type Input = Input0

  override protected def rawForward(input: BatchId.Aux[Input]): SharedBatch = {
    val upstream = operand.forward(input).open()
    new SharedBatch(input, upstream)
  }
}
