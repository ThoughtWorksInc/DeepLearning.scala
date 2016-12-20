package com.thoughtworks.deeplearning.dsl.layers

import com.thoughtworks.deeplearning.{Batch, Layer}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Throw(throwable: () => Throwable) extends Layer with Batch {
  override type Input = Batch
  override type Output = Batch.Aux[scala.Nothing, scala.Any]
  override type Data = scala.Nothing
  override type Delta = scala.Any

  override def forward(input: Input) = this

  override def backward(delta: Delta): Unit = {}

  override def value: Data = {
    throw throwable()
  }

  override def close(): Unit = {}

  override def addReference() = this
}
