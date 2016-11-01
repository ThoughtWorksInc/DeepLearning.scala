package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.Ast._
import com.thoughtworks.deepLearning.Batch._
import org.nd4j.linalg.api.ndarray.INDArray
import com.thoughtworks.deepLearning.array2D.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Sum[Input0 <: Batch](operand: WidenAst[Input0, WidenBatch[Eval[INDArray], Eval[INDArray]]],
                                      dimensions: Seq[Int])
    extends Cached {

  protected final class SharedBatch(override val input: Input0, upstream: WidenBatch[Eval[INDArray], Eval[INDArray]])
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
