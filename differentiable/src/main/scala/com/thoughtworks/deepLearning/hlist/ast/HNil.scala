package com.thoughtworks.deepLearning.hlist.ast

import com.thoughtworks.deepLearning.Ast._
import com.thoughtworks.deepLearning.Batch._
import com.thoughtworks.deepLearning.{Batch, Ast}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
case object HNil extends Ast with Batch {
  override type Input = Batch

  override type Data = shapeless.HNil

  override type Delta = shapeless.CNil

  override type Output = WidenBatch[Data, Delta]

  override def forward(input: Input): Output = this

  override def backward(delta: Delta): Unit = {}

  override def value = shapeless.HNil

  override def close(): Unit = {}
}
