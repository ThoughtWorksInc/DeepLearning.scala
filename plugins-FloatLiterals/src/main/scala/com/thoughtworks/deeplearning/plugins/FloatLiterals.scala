package com.thoughtworks.deeplearning
package plugins

import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.raii.asynchronous.Do

import com.thoughtworks.future.continuation.Continuation

/** A plugin that enables [[scala.Float]] in neural networks. */
trait FloatLiterals {

  trait ImplicitsApi {
    implicit def floatLiteralDeepLearning: DeepLearning.Aux[Float, Float, Float] = new DeepLearning[Float] {
      override type Data = Float
      override type Delta = Float

      override def forward(literal: Float): Do[Tape[Data, Delta]] = {
        Do.now(Tape(literal, Function.const(Continuation.now(()))))
      }
    }
  }

  type Implicits <: ImplicitsApi
}
