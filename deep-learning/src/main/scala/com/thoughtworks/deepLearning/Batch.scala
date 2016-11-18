package com.thoughtworks.deepLearning

import scala.annotation.elidable

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object Batch {

  /** @template */
  type Aux[+Data0, -Delta0] = Batch {
    type Data <: Data0
    type Delta >: Delta0
  }

}

trait Batch extends AutoCloseable {
  type Data
  type Delta

  def backward(delta: Delta): Unit

  def value: Data
}
