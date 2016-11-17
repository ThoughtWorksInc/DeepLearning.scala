package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.NeuralNetwork._
import com.thoughtworks.deepLearning.Batch._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning.array2D.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Log[Input0 <: Batch](operand: NeuralNetwork.Aux[Input0, Array2D#ConcreteBatch])
    extends Cached {

  protected final class SharedBatch private[deepLearning](override val input: BatchId.Aux[Input0], upstream: Array2D#ConcreteBatch)
      extends Array2DSemigroupBatch
      with SemigroupBatch {
    val value = upstream.value.map(Transforms.log).memoize

    type Input >: Input0

    override protected def closeUpstreams(): Unit = {
      upstream.close()
    }

    override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
      upstream.backward(outputDelta.map2(upstream.value)(_ / _).memoize)
    }
  }

  type Input = Input0

  override protected def rawForward(input: BatchId.Aux[Input]): SharedBatch = {
    val upstream = operand.forward(input).open()
    new SharedBatch(input, upstream)
  }
}
