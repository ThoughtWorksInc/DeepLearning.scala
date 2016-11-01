package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import com.thoughtworks.deepLearning.Ast._
import com.thoughtworks.deepLearning.Batch._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning.array2D.utilities.Array2DSemigroupBatch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Negative[Input0 <: Batch](operand: WidenAst[Input0, WidenBatch[Eval[INDArray], Eval[INDArray]]])
    extends Cached {

  protected final class SharedBatch private[deepLearning](override val input: Input0, upstream: WidenBatch[Eval[INDArray], Eval[INDArray]])
      extends Array2DSemigroupBatch
      with SemigroupBatch {
    val value = upstream.value.map(-_).memoize

    type Input >: Input0

    override protected def closeUpstreams(): Unit = {
      upstream.close()
    }

    override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
      upstream.backward(outputDelta.map(-_).memoize)
    }
  }

  type Input = Input0

  override protected def rawForward(input: Input): SharedBatch = {
    val upstream = operand.forward(input)
    new SharedBatch(input, upstream)
  }
}
