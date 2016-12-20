package com.thoughtworks.deeplearning.dsl.layers

import com.thoughtworks.deeplearning.{Batch, Layer}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Literal[Data0](value0: Data0) extends Layer with Batch {
  override type Data = Data0
  override type Delta = scala.Any
  override type Input = Batch
  override type Output = Batch.Aux[Data, Delta]

  override def value: Data = value0

  override def forward(input: Input) = this

  override def backward(delta: Delta): Unit = {}

  override def close(): Unit = {}

  override def addReference() = this
}
