package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.Differentiable._
import com.thoughtworks.deepLearning.Differentiable.Batch
import shapeless.DepFn1

import scala.language.higherKinds
import scalaz.Liskov
import scalaz.Liskov.<~<

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

  /**
    * @note This is a workaround for https://issues.scala-lang.org/browse/SI-10008
    * @template
    */
  type Batch >: Differentiable.Batch[Data, Delta] <: Differentiable.Batch[Data, Delta]

  /**
    * @note This is a workaround for https://issues.scala-lang.org/browse/SI-10008
    * @template
    */
  type Ast[Output <: Differentiable] >: DifferentiableFunction.Ast[Batch, Output#Batch] <: DifferentiableFunction.Ast[Batch, Output#Batch]

  final def toBatch: Batch = this: Differentiable.Batch[Data, Delta]

  def backward(delta: Delta): Unit

  def value: Data
}
