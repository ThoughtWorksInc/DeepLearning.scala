package com.thoughtworks.deeplearning
package double.layers

import cats._
import cats.implicits._
import com.thoughtworks.deeplearning.BufferedLayer
import com.thoughtworks.deeplearning.double.utilities.DoubleMonoidBatch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Times[Input0 <: Batch](
    operand1: Layer.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
    operand2: Layer.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
) extends BufferedLayer.Binary {

  type BufferedBatch = DoubleMonoidBatch with MonoidBatch with BinaryBatch

  type Input = Input0

  override protected def rawForward(input0: Input): BufferedBatch = {
    new {
      override final val input = input0
    } with  DoubleMonoidBatch with MonoidBatch with BinaryBatch {

      override final val value = upstream1.value.map2(upstream2.value)(_ * _)

      override protected def rawBackward(delta: Eval[scala.Double]): Unit = {
        val a = upstream1.value
        val b = upstream2.value
        upstream1.backward(delta.map2(b)(_ * _))
        upstream2.backward(delta.map2(a)(_ * _))
      }

    }
  }
}
