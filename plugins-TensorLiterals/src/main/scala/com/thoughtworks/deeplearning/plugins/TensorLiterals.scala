package com.thoughtworks.deeplearning
package plugins

import com.thoughtworks.compute.Tensors
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.raii.asynchronous.Do

import com.thoughtworks.continuation.Continuation

/**
  * @author 杨博 (Yang Bo)
  */
trait TensorLiterals extends Tensors {

  implicit def tensorLiteralDeepLearning: DeepLearning.Aux[Tensor, Tensor, Tensor] = new DeepLearning[Tensor] {

    override type Data = Tensor
    override type Delta = Tensor

    override def forward(literal: Tensor): Do[Tape[Data, Delta]] = {
      Do.now(Tape(literal, Function.const(Continuation.now(()))))
    }
  }

}
