package com.thoughtworks.deepLearning
package array2D.layer

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.Layer._
import com.thoughtworks.deepLearning.Batch._
import com.thoughtworks.deepLearning.array2D.utilities._
import com.thoughtworks.deepLearning.double.utilities.DoubleMonoidBatch
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class ReduceSum[Input0 <: Batch](operand: Layer.Aux[Input0, Array2D#Batch]) extends BufferedLayer {

  protected final class BufferedBatch private[deepLearning] (override val input: BatchId.Aux[Input0],
                                                             upstream: Array2D#Batch)
      extends MonoidBatch
      with DoubleMonoidBatch {

    val value = upstream.value.map(_.sumT).memoize

    override protected def closeUpstreams(): Unit = {
      upstream.close()
    }

    override protected def rawBackward(outputDelta: Eval[scala.Double]): Unit = {
      upstream.backward(
        outputDelta
          .map2(upstream.value) { (outputDeltaValue, aValue) =>
            Nd4j.valueArrayOf(aValue.shape(), outputDeltaValue)
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
