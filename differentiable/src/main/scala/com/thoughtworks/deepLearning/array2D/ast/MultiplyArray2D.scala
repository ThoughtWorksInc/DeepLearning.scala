package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.Ast._
import com.thoughtworks.deepLearning.Batch._
import org.nd4j.linalg.api.ndarray.INDArray
import com.thoughtworks.deepLearning.array2D.utilities._
import org.nd4s.Implicits._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class MultiplyArray2D[Input0 <: Batch](
    leftOperand: WidenAst[Input0, WidenBatch[Eval[INDArray], Eval[INDArray]]],
    rightOperand: WidenAst[Input0, WidenBatch[Eval[INDArray], Eval[INDArray]]]
) extends Ast
    with Cached {

  protected final class SharedBatch private[deepLearning](override val input: Input0,
                                    upstream1: WidenBatch[Eval[INDArray], Eval[INDArray]],
                                    upstream2: WidenBatch[Eval[INDArray], Eval[INDArray]])
      extends Array2DSemigroupBatch
      with SemigroupBatch {
    val value = {
      upstream1.value
        .map2(upstream2.value) { (aValue, bValue) =>
          val Array(aRows, aColumns) = aValue.shape()
          val Array(bRows, bColumns) = bValue.shape()
          val newShape =
            Array(math.max(aRows, bRows), math.max(aColumns, bColumns))
          aValue.broadcast(newShape: _*) * bValue.broadcast(newShape: _*)
        }
        .memoize
    }

    type Input >: Input0

    override protected def closeUpstreams(): Unit = {
      upstream1.close()
      upstream2.close()
    }

    override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
      val a = upstream1.value
      val b = upstream2.value
      upstream1.backward(
        Applicative[Eval]
          .map3(outputDelta, a, b) { (outputDeltaValue, aData, bData) =>
            sumAs(bData.broadcast(outputDeltaValue.shape(): _*) * outputDeltaValue, aData.shape())
          }
          .memoize)
      upstream2.backward(
        Applicative[Eval]
          .map3(outputDelta, a, b) { (outputDeltaValue, aData, bData) =>
            sumAs(aData.broadcast(outputDeltaValue.shape(): _*) * outputDeltaValue, bData.shape())
          }
          .memoize)
    }
  }

  type Input = Input0

  override protected def rawForward(input: Input): SharedBatch = {
    new SharedBatch(input, leftOperand.forward(input), rightOperand.forward(input))
  }
}
