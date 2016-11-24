package com.thoughtworks.deeplearning
package double.layers

import cats._
import cats.implicits._

import com.thoughtworks.deeplearning.Layer._

import com.thoughtworks.deeplearning.double.utilities.DoubleMonoidBatch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Log[Input0 <: Batch](operand: Layer.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]])
    extends BufferedLayer {

  protected final class BufferedBatch private[deeplearning] (
      override val input: BatchId.Aux[Input0],
      upstream: Batch.Aux[Eval[scala.Double], Eval[scala.Double]])
      extends MonoidBatch
      with DoubleMonoidBatch {

    val value = upstream.value.map(math.log).memoize

    override protected def closeUpstreams(): Unit = {
      upstream.close()
    }

    override protected def rawBackward(outputDelta: Eval[scala.Double]): Unit = {
      upstream.backward(outputDelta.map2(upstream.value)(_ / _).memoize)
    }
  }

  type Input = Input0

  override protected def rawForward(input: BatchId.Aux[Input]): BufferedBatch = {
    val upstream = operand.forward(input).open()
    new BufferedBatch(input, upstream)
  }
}
