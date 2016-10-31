package com.thoughtworks.deepLearning

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object Batch {
  type Aux[+Data0, -Delta0] = Batch {
    type Data <: Data0
    type Delta >: Delta0
  }

  type FromTypePair[TypePair <: { type Data; type Delta }] = Batch.Aux[TypePair#Data, TypePair#Delta]
}

trait Batch extends AutoCloseable {
  type Data
  type Delta

  def backward(delta: Delta): Unit

  def value: Data
}
