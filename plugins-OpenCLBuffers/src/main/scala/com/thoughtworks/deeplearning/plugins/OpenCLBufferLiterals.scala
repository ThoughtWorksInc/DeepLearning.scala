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
    implicit def indArrayLiteralDeepLearning[Element]
      : DeepLearning.Aux[DeviceBuffer[Element], DeviceBuffer[Element], DeviceBuffer[Element]] =
      new DeepLearning[DeviceBuffer[Element]] {
        override type Data = DeviceBuffer[Element]
        override type Delta = DeviceBuffer[Element]

        override def forward(literal: DeviceBuffer[Element]): Do[Tape[Data, Delta]] = {
          Do.now(Tape(literal, Function.const(Continuation.now(()))))
        }
      }
  }

  type Implicits <: ImplicitsApi
}
