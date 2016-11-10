package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.Differentiable._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning.array2D.utilities._
import com.thoughtworks.deepLearning.double.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class MultiplyDouble[Input0 <: Differentiable](
                                                           leftOperand: DifferentiableFunction.Ast[Input0, Array2D#ConcreteBatch],
                                                           rightOperand: DifferentiableFunction.Ast[Input0, Double#ConcreteBatch]
) extends DifferentiableFunction
    with Cached {

  protected final class SharedBatch private[deepLearning](override val input: Input0,
                                                          upstream1: Array2D#ConcreteBatch,
                                                          upstream2: Double#ConcreteBatch)
      extends Array2DSemigroupBatch
      with SemigroupBatch {
    val value = upstream1.value.map2(upstream2.value)(_ * _).memoize

    type Input >: Input0

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

  override protected def rawForward(input: Input): SharedBatch = {
    new SharedBatch(input, leftOperand.forward(input), rightOperand.forward(input))
  }
}
