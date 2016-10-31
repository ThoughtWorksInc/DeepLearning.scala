package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import com.thoughtworks.deepLearning.Ast
import com.thoughtworks.deepLearning.Ast._
import org.nd4j.linalg.api.ndarray.INDArray
import com.thoughtworks.deepLearning.array2D.utilities._
import org.nd4s.Implicits._
import cats.implicits._
import com.thoughtworks.deepLearning.Ast.Cached
import com.thoughtworks.deepLearning.array2D.utilities.Array2DSemigroupBatch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class AddDouble[Input0 <: Batch](
                                             leftOperand: Ast.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]],
                                             rightOperand: Ast.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
) extends Ast
    with Cached {

  protected final class SharedBatch(override val input: Input0,
                                    upstream1: Batch.Aux[Eval[INDArray], Eval[INDArray]],
                                    upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]])
      extends Array2DSemigroupBatch
      with SemigroupBatch {
    val value = upstream1.value.map2(upstream2.value)(_ + _).memoize

    type Input >: Input0

    override protected def closeUpstreams(): Unit = {
      upstream1.close()
      upstream2.close()
    }

    override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
      upstream1.backward(outputDelta)
      upstream2.backward(outputDelta.map(_.sumT))
    }
  }

  type Input = Input0

  override protected def rawForward(input: Input): SharedBatch = {
    new SharedBatch(input, leftOperand.forward(input), rightOperand.forward(input))
  }
}
