package com.thoughtworks.deepLearning
package double.ast

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.BufferedNetwork
import com.thoughtworks.deepLearning.boolean.utilities.BooleanMonoidBatch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class LessThan[Input0 <: Batch](
    leftOperand: NeuralNetwork.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
    rightOperand: NeuralNetwork.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
) extends BufferedNetwork {

  protected final class BufferedBatch private[deepLearning] (
      override val input: BatchId.Aux[Input0],
      upstream1: Batch.Aux[Eval[scala.Double], Eval[scala.Double]],
      upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]])
      extends MonoidBatch
      with BooleanMonoidBatch {

    val value = upstream1.value.map2(upstream2.value)(_ < _).memoize

    override protected def closeUpstreams(): Unit = {
      upstream1.close()
      upstream2.close()
    }

    override protected def rawBackward(delta: Eval[scala.Boolean]): Unit = {
      upstream1.backward(Eval.now(0.0))
      upstream2.backward(Eval.now(0.0))
    }
  }

  type Input = Input0

  override protected def rawForward(input: BatchId.Aux[Input]): BufferedBatch = {
    new BufferedBatch(input, leftOperand.forward(input).open(), rightOperand.forward(input).open())
  }
}
