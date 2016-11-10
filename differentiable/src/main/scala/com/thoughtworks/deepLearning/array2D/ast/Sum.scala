package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.Differentiable._
import org.nd4j.linalg.api.ndarray.INDArray
import com.thoughtworks.deepLearning.array2D.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Sum[Input0 <: Differentiable](operand: DifferentiableFunction.Ast[Input0, Array2D#ConcreteBatch],
                                               dimensions: Seq[Int])
    extends Cached {

  protected final class SharedBatch private[deepLearning](override val input: Input0, upstream: Array2D#ConcreteBatch)
      extends Array2DSemigroupBatch
      with SemigroupBatch {
    val value = upstream.value.map(_.sum(dimensions: _*)).memoize

    type Input >: Input0

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

  override protected def rawForward(input: Input): SharedBatch = {
    val upstream = operand.forward(input)
    new SharedBatch(input, upstream)
  }
}
