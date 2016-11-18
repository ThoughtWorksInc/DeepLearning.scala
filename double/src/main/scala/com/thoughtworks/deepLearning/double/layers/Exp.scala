package com.thoughtworks.deepLearning
package double.layers

import cats._
import cats.implicits._

import com.thoughtworks.deepLearning.Layer._

import com.thoughtworks.deepLearning.double.utilities.DoubleMonoidBatch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Exp[Input0 <: Batch](operand: Layer.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]])
    extends BufferedLayer {

  protected final class BufferedBatch private[deepLearning] (
      override val input: BatchId.Aux[Input0],
      upstream: Batch.Aux[Eval[scala.Double], Eval[scala.Double]])
      extends MonoidBatch
      with DoubleMonoidBatch {

    val value = upstream.value.map(math.exp).memoize

    override protected def closeUpstreams(): Unit = {
      upstream.close()
    }

    override protected def rawBackward(outputDelta: Eval[scala.Double]): Unit = {
      upstream.backward(value.map2(outputDelta)(_ * _).memoize)
    }
  }

  type Input = Input0

  override protected def rawForward(input: BatchId.Aux[Input]) = {
    new BufferedBatch(input, operand.forward(input).open())
  }
}
