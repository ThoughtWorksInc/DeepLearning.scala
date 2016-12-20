package com.thoughtworks.deeplearning.double.layers

import cats._
import cats.implicits._
import com.thoughtworks.deeplearning.double.optimizers.Optimizer
import com.thoughtworks.deeplearning.{Batch, Layer}
import com.thoughtworks.deeplearning.double.utilities.DoubleMonoidBatch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Weight(var rawValue: Double)(implicit optimizer: Optimizer)
    extends Layer
    with DoubleMonoidBatch {
  override type Input = Batch
  override type Output = Batch.Aux[Data, Delta]

  override def addReference() = this

  override def forward(any: Input) = this

  override def backward(delta: Delta): Unit = {
    synchronized {
      rawValue = optimizer.updateDouble(rawValue, delta.value)
    }
  }

  override def value = Eval.now(rawValue)

  override def close(): Unit = {}

}
