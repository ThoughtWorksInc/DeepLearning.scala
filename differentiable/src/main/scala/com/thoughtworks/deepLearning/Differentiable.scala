package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.Differentiable._
import shapeless.DepFn1

import scala.language.higherKinds

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object Differentiable {

  type Aux[Data0, Delta0] = Differentiable {
    type Data = Data0
    type Delta = Delta0
  }

  /** @template */
  type Batch[+Data0, -Delta0] = Differentiable {
    type Data <: Data0
    type Delta >: Delta0
  }

}

trait Differentiable extends AutoCloseable { outer =>
  type Data
  type Delta

  def backward(delta: Delta): Unit

  def value: Data
}
