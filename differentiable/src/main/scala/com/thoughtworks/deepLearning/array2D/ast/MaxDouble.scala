package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import com.thoughtworks.deepLearning.Differentiable
import com.thoughtworks.deepLearning.Differentiable._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import com.thoughtworks.deepLearning.array2D.utilities._
import org.nd4s.Implicits._
import cats.implicits._
import com.thoughtworks.deepLearning.any.utilities.Cached
import com.thoughtworks.deepLearning.array2D.utilities.Array2DSemigroupBatch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class MaxDouble[Input0 <: Batch](
    leftOperand: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]],
    rightOperand: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
) extends Differentiable
    with Cached {

  protected final class SharedBatch(override val input: Input0,
                                    upstream1: Batch.Aux[Eval[INDArray], Eval[INDArray]],
                                    upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]])
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
