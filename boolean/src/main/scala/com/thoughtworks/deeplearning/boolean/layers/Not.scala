package com.thoughtworks.deeplearning.boolean.layers

import cats._
import com.thoughtworks.deeplearning.{Batch, BatchId, Layer}
import com.thoughtworks.deeplearning.BufferedLayer
import com.thoughtworks.deeplearning.boolean.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Not[Input0 <: Batch](differentiableBoolean: Layer.Aux[Input0, Boolean#Batch]) extends BufferedLayer {

  protected final class BufferedBatch private[deeplearning] (override val input: BatchId.Aux[Input0],
                                                             upstream: Boolean#Batch)
      extends MonoidBatch
      with BooleanMonoidBatch {

    val value = upstream.value.map(!_)

    override protected def rawBackward(delta: Eval[scala.Boolean]): Unit = {
      upstream.backward(delta.map(!_))
    }

    override protected def closeUpstreams(): Unit = {
      upstream.close()
    }
  }

  type Input = Input0

  override protected def rawForward(input: BatchId.Aux[Input0]): BufferedBatch = {
    val upstream = differentiableBoolean.forward(input)
    new BufferedBatch(input, upstream.open())
  }
}
