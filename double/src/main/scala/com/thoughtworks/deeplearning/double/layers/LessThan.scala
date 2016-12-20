package com.thoughtworks.deeplearning
package double.layers

import cats._
import cats.implicits._
import com.thoughtworks.deeplearning.BufferedLayer
import com.thoughtworks.deeplearning.BpBoolean.BooleanMonoidBatch
import com.thoughtworks.deeplearning.Layer.Batch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class LessThan[Input0 <: Batch](
    operand1: Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]],
    operand2: Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]]
) extends BufferedLayer.Binary {

  type BufferedBatch = BooleanMonoidBatch with MonoidBatch with BinaryBatch

  type Input = Input0

  override protected def rawForward(input0: Input): BufferedBatch = {
    new {
      override final val input = input0
    } with BooleanMonoidBatch with MonoidBatch with BinaryBatch {
      override val value = upstream1.value.map2(upstream2.value)(_ < _).memoize

      override protected def rawBackward(delta: Eval[Boolean]): Unit = {
        upstream1.backward(Eval.now(0.0))
        upstream2.backward(Eval.now(0.0))
      }
    }
  }
}
