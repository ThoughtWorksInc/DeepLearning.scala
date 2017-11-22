package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.compute.OpenCL
import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.deeplearning.plugins.Layers._

trait DeviceBufferLayers extends Layers with OpenCL {

  trait ImplicitsApi extends super[Layers].ImplicitsApi with super[OpenCL].ImplicitsApi
  override type Implicits <: ImplicitsApi

  trait DeviceBufferLayerApi extends LayerApi {

    override type Data = DeviceBuffer[Element]
    override type Delta = DeviceBuffer[Element]
    type Element
  }

  type DeviceBufferLayer <: DeviceBufferLayerApi with Layer
  def mean[Operand0, Buffer, Element, OutputLayer](operand0: Operand0)(
      implicit
      deepLearning0: DeepLearning.Aux[Operand0, Buffer, Buffer],
      isDoBuffer: Do[Tape[Buffer, Buffer]] <:< Do[Tape[DeviceBuffer[Element], DeviceBuffer[Element]]],
      layerFactory: ToLayer.Aux[Element, Element, OutputLayer]): OutputLayer = ???

  def matrixMultiply[Operand0, Operand1, Buffer, Element, OutputLayer /* <: DeviceBufferLayer */ ](operand0: Operand0,
                                                                                                   operand1: Operand1,
                                                                                                   length: Int)(
      implicit
      deepLearning0: DeepLearning.Aux[Operand0, Buffer, Buffer],
      deepLearning1: DeepLearning.Aux[Operand1, Buffer, Buffer],
      isDoBuffer: Do[Tape[Buffer, Buffer]] <:< Do[Tape[DeviceBuffer[Element], DeviceBuffer[Element]]],
      layerFactory: ToLayer.Aux[DeviceBuffer[Element], DeviceBuffer[Element], OutputLayer]
  ): OutputLayer = ???

  def subtract[Buffer, Element, Operand0, Operand1, OutputLayer /* <: DeviceBufferLayer */ ](operand0: Operand0,
                                                                                             operand1: Operand1)(
      implicit
      deepLearning0: DeepLearning.Aux[Operand0, Buffer, Buffer],
      deepLearning1: DeepLearning.Aux[Operand1, Buffer, Buffer],
      isDoBuffer: Do[Tape[Buffer, Buffer]] <:< Do[Tape[DeviceBuffer[Element], DeviceBuffer[Element]]],
      layerFactory: ToLayer.Aux[DeviceBuffer[Element], DeviceBuffer[Element], OutputLayer]
  ): OutputLayer = {
    val do0: Do[Tape[DeviceBuffer[Element], DeviceBuffer[Element]]] = isDoBuffer(deepLearning0.forward(operand0))

    ???
  }

  def multiply[Buffer, Element, Operand0, Operand1, OutputLayer /* <: DeviceBufferLayer */ ](operand0: Operand0,
                                                                                             operand1: Operand1)(
      implicit
      deepLearning0: DeepLearning.Aux[Operand0, Buffer, Buffer],
      deepLearning1: DeepLearning.Aux[Operand1, Buffer, Buffer],
      isDoBuffer: Do[Tape[Buffer, Buffer]] <:< Do[Tape[DeviceBuffer[Element], DeviceBuffer[Element]]],
      layerFactory: ToLayer.Aux[DeviceBuffer[Element], DeviceBuffer[Element], OutputLayer]
  ): OutputLayer = ???

}
