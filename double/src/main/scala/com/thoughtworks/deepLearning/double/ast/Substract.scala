package com.thoughtworks.deepLearning
package double.ast

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.NeuralNetwork._
import com.thoughtworks.deepLearning.double.utilities.DoubleMonoidBatch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Substract[Input0 <: Batch](
                                             leftOperand: NeuralNetwork.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
                                             rightOperand: NeuralNetwork.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
) extends Cached {

  protected final class SharedBatch private[deepLearning] (
                                                            override val input: BatchId.Aux[Input0],
                                                            upstream1: Batch.Aux[Eval[scala.Double], Eval[scala.Double]],
                                                            upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]])
      extends MonoidBatch
      with DoubleMonoidBatch {
    type Input >: Input0
    val value = upstream1.value.map2(upstream2.value)(_ - _)

    override protected def closeUpstreams(): Unit = {
      upstream1.close()
      upstream2.close()
    }

    override protected def rawBackward(delta: Eval[scala.Double]): Unit = {
      upstream1.backward(delta)
      upstream2.backward(delta.map(-_))
    }
  }

  type Input = Input0

  override protected def rawForward(input: BatchId.Aux[Input]): SharedBatch = {
    new SharedBatch(input, leftOperand.forward(input).open(), rightOperand.forward(input).open())
  }

}
