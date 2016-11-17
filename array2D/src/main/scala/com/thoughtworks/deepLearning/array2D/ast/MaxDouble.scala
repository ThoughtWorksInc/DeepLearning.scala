package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import com.thoughtworks.deepLearning.NeuralNetwork._
import com.thoughtworks.deepLearning.Batch._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import cats.implicits._
import com.thoughtworks.deepLearning.NeuralNetwork.Cached
import com.thoughtworks.deepLearning.array2D.utilities._
import com.thoughtworks.deepLearning.double.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class MaxDouble[Input0 <: Batch](
                                                      leftOperand: NeuralNetwork.Aux[Input0, Array2D#ConcreteBatch],
                                                      rightOperand: NeuralNetwork.Aux[Input0, Double#ConcreteBatch]
) extends NeuralNetwork
    with Cached {

  protected final class SharedBatch private[deepLearning] (override val input: Input0,
                                                           upstream1: Array2D#ConcreteBatch,
                                                           upstream2: Double#ConcreteBatch)
      extends Array2DSemigroupBatch
      with SemigroupBatch {
    val value = upstream1.value.map2(upstream2.value)(Transforms.max).memoize

    type Input >: Input0

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

  override protected def rawForward(input: Input): SharedBatch = {
    new SharedBatch(input, leftOperand.forward(input), rightOperand.forward(input))
  }
}
