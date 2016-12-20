package com.thoughtworks.deeplearning
package hlist.layers

import com.thoughtworks.deeplearning.Layer.Batch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
case object HNil extends Layer with Batch {
  override type Input = Batch

  override type Data = shapeless.HNil

  override type Delta = shapeless.CNil

  override type Output = Batch.Aux[Data, Delta]

  override def addReference() = this

  override def forward(input: Input) = this

  override def backward(delta: Delta): Unit = {}

  override def value = shapeless.HNil

  override def close(): Unit = {}
}
