package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.Differentiable._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning.array2D.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Dot[Input0 <: Differentiable](
                                                leftOperand: DifferentiableFunction.Ast[Input0, Array2D#Batch],
                                                rightOperand: DifferentiableFunction.Ast[Input0, Array2D#Batch]
) extends DifferentiableFunction
    with Cached {

  protected final class SharedBatch private[deepLearning](override val input: Input0,
                                                          upstream1: Array2D#Batch,
                                                          upstream2: Array2D#Batch)
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
