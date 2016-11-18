package com.thoughtworks.deepLearning
package array2D.layers

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.Layer._
import com.thoughtworks.deepLearning.Batch._
import org.nd4j.linalg.api.ndarray.INDArray
import com.thoughtworks.deepLearning.array2D.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Sum[Input0 <: Batch](operand: Layer.Aux[Input0, Array2D#Batch], dimensions: Seq[Int])
    extends BufferedLayer {

  protected final class BufferedBatch private[deepLearning] (override val input: BatchId.Aux[Input0],
                                                             upstream: Array2D#Batch)
      extends Array2DSemigroupBatch
      with SemigroupBatch {
    val value = upstream.value.map(_.sum(dimensions: _*)).memoize

    override protected def closeUpstreams(): Unit = {
      upstream.close()
    }

    override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
      val a = upstream.value
      upstream.backward(
        outputDelta
          .map2(a) { (outputDeltaValue, aValue) =>
            outputDeltaValue.broadcast(aValue.shape: _*)
          }
          .memoize)
    }
  }

  type Input = Input0

  override protected def rawForward(input: BatchId.Aux[Input]): BufferedBatch = {
    val upstream = operand.forward(input).open()
    new BufferedBatch(input, upstream)
  }
}
