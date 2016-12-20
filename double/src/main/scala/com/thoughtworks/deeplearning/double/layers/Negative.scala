package com.thoughtworks.deeplearning.double.layers

import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.Batch._
import cats._
import cats.implicits._

import com.thoughtworks.deeplearning._
import com.thoughtworks.deeplearning.BufferedLayer
import com.thoughtworks.deeplearning.double.utilities.DoubleMonoidBatch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Negative[Input0 <: Batch](
    operand: Layer.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]])
    extends BufferedLayer.Unary {

  type BufferedBatch = DoubleMonoidBatch with MonoidBatch with UnaryBatch

  type Input = Input0

  override protected def rawForward(input0: Input) = new {
    override final val input = input0
  } with  MonoidBatch with DoubleMonoidBatch with UnaryBatch {

    val value = upstream.value.map(-_)

    override protected def rawBackward(delta: Eval[scala.Double]): Unit = {
      upstream.backward(delta.map(-_))
    }

  }

}
