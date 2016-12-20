package com.thoughtworks.deeplearning.boolean.layers

import cats._
import com.thoughtworks.deeplearning.{Batch, Layer}
import com.thoughtworks.deeplearning.BufferedLayer
import com.thoughtworks.deeplearning.boolean.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Not[Input0 <: Batch](operand: Layer.Aux[Input0, Boolean#Batch]) extends BufferedLayer.Unary {

  type BufferedBatch = MonoidBatch with BooleanMonoidBatch with UnaryBatch

  type Input = Input0

  override protected def rawForward(input0: Input): BufferedBatch = {
    new {

      override val input = input0

    } with MonoidBatch with BooleanMonoidBatch with UnaryBatch {

      override val value = upstream.value.map(!_)

      override protected def rawBackward(delta: Eval[scala.Boolean]): Unit = {
        upstream.backward(delta.map(!_))
      }

    }
  }
}
