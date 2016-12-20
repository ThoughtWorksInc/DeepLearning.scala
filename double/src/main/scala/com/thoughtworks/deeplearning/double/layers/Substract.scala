package com.thoughtworks.deeplearning
package double.layers

import cats._
import cats.implicits._
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.double.utilities.DoubleMonoidBatch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Substract[Input0 <: Batch](
    operand1: Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]],
    operand2: Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]]
) extends BufferedLayer.Binary {

  type BufferedBatch = DoubleMonoidBatch with MonoidBatch with BinaryBatch

  type Input = Input0

  override protected def rawForward(input0: Input): BufferedBatch = {
    new {
      override final val input = input0
    } with  DoubleMonoidBatch with MonoidBatch with BinaryBatch {

      val value = upstream1.value.map2(upstream2.value)(_ - _)

      override protected def rawBackward(delta: Eval[Double]): Unit = {
        upstream1.backward(delta)
        upstream2.backward(delta.map(-_))
      }

    }
  }
}
