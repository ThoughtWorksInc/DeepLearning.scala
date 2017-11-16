package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.compute.OpenCL
import com.thoughtworks.continuation.Continuation
import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.raii.asynchronous.Do

/**
  * @author 杨博 (Yang Bo)
  */
trait OpenCLBufferLiterals extends OpenCL {

  trait ImplicitsApi extends super.ImplicitsApi {
    implicit def indArrayLiteralDeepLearning
      : DeepLearning.Aux[DeviceBuffer[Float], DeviceBuffer[Float], DeviceBuffer[Float]] =
      new DeepLearning[DeviceBuffer[Float]] {
        override type Data = DeviceBuffer[Float]
        override type Delta = DeviceBuffer[Float]

        override def forward(literal: DeviceBuffer[Float]): Do[Tape[Data, Delta]] = {
          Do.now(Tape(literal, Function.const(Continuation.now(()))))
        }
      }
  }

  type Implicits <: ImplicitsApi
}
