package com.thoughtworks.deepLearning.boolean.ast

import cats._
import com.thoughtworks.deepLearning.Differentiable
import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.DifferentiableFunction.Cached
import com.thoughtworks.deepLearning.boolean.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Not[Input0 <: Differentiable](differentiableBoolean: Ast[Input0, Boolean#Widen]) extends Cached {

  protected final class SharedBatch private[deepLearning] (override val input: Input0, upstream: Boolean#Widen)
      extends MonoidBatch
      with BooleanMonoidBatch {
    type Input >: Input0
    val value = upstream.value.map(!_)

    override protected def rawBackward(delta: Eval[scala.Boolean]): Unit = {
      upstream.backward(delta.map(!_))
    }

    override protected def closeUpstreams(): Unit = {
      upstream.close()
    }
  }

  type Input = Input0

  override protected def rawForward(input: Input): SharedBatch = {
    val upstream = differentiableBoolean.forward(input)
    new SharedBatch(input, upstream)
  }
}
