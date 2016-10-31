package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import com.thoughtworks.deepLearning.Differentiable
import com.thoughtworks.deepLearning.Differentiable._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import com.thoughtworks.deepLearning.array2D.utilities._
import cats.implicits._
import com.thoughtworks.deepLearning.any.utilities.Cached
import com.thoughtworks.deepLearning.array2D.utilities.Array2DSemigroupBatch
import org.nd4s.Implicits._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Exp[Input0 <: Batch](operand: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]])
    extends Cached {

  protected final class SharedBatch(override val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[INDArray]])
      extends Array2DSemigroupBatch
      with SemigroupBatch {
    val value = upstream.value.map(Transforms.exp).memoize

    type Input >: Input0

    override protected def closeUpstreams(): Unit = {
      upstream.close()
    }

    override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
      upstream.backward(value.map2(outputDelta)(_ * _).memoize)
    }
  }

  type Input = Input0

  override protected def rawForward(input: Input): SharedBatch = {
    val upstream = operand.forward(input)
    new SharedBatch(input, upstream)
  }
}
