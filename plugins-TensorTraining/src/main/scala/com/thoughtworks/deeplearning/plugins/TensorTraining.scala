package com.thoughtworks.deeplearning.plugins

import algebra.ring.MultiplicativeMonoid
import com.thoughtworks.compute.Tensors

/** A DeepLearning.scala plugin that enables [[DeepLearning.Ops.train train]] method for neural networks whose loss is a [[Tensor]].
  *
  * @author 杨博 (Yang Bo)
  */
trait TensorTraining extends Training with Tensors {

  private lazy val One: Tensor = Tensor.scalar(1.0f)

  trait ImplicitsApi extends super.ImplicitsApi {
    implicit final def tensorMultiplicativeMonoid: MultiplicativeMonoid[Tensor] =
      new MultiplicativeMonoid[Tensor] {
        override def one: Tensor = One

        override def times(x: Tensor, y: Tensor): Tensor = x * y
      }
  }
  type Implicits <: ImplicitsApi
}
