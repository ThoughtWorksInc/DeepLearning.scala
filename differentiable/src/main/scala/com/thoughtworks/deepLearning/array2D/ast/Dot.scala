package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.Differentiable
import com.thoughtworks.deepLearning.Differentiable._
import com.thoughtworks.deepLearning.any.utilities.Cached
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning.array2D.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Dot[Input0 <: Batch](
    leftOperand: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]],
    rightOperand: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]]
) extends Differentiable
    with Cached {

  protected final class SharedBatch(override val input: Input0,
                                    upstream1: Batch.Aux[Eval[INDArray], Eval[INDArray]],
                                    upstream2: Batch.Aux[Eval[INDArray], Eval[INDArray]])
      extends Array2DSemigroupBatch
      with SemigroupBatch {
    override val value = upstream1.value.map2(upstream2.value)(_ dot _).memoize

    type Input >: Input0

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

  override protected def rawForward(input: Input): SharedBatch = {
    new SharedBatch(input, leftOperand.forward(input), rightOperand.forward(input))
  }
}
