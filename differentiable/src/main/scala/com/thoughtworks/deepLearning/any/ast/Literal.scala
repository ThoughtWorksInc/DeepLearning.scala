package com.thoughtworks.deepLearning.any.ast

import com.thoughtworks.deepLearning.{Batch, Ast}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Literal[Data0](value0: Data0) extends Ast with Batch {
  override type Data = Data0
  override type Delta = scala.Any
  override type Input = Batch
  override type Output = Batch.Aux[Data, Delta]

  override def value: Data = value0

  override def forward(input: Input): Output = this

  override def backward(delta: Delta): Unit = {}

  override def close(): Unit = {}
}
