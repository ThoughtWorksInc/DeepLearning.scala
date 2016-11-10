package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.Differentiable._
import com.thoughtworks.deepLearning.array2D.utilities._
import com.thoughtworks.deepLearning.double.utilities.DoubleMonoidBatch
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class ReduceSum[Input0 <: Differentiable](operand: DifferentiableFunction.Ast[Input0, Array2D#ConcreteBatch])
    extends Cached {

  protected final class SharedBatch private[deepLearning](override val input: Input0, upstream: Array2D#ConcreteBatch)
      extends MonoidBatch
      with DoubleMonoidBatch {
    type Input >: Input0
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

  override protected def rawForward(input: Input): SharedBatch = {
    val upstream = operand.forward(input)
    new SharedBatch(input, upstream)
  }
}
