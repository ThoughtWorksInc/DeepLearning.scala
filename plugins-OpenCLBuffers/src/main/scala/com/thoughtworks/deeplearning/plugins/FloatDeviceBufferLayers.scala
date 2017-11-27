package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.deeplearning.plugins.Layers.ToLayer
import com.thoughtworks.deeplearning.plugins._
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply}
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.raii.asynchronous.Do

trait FloatDeviceBufferLayers extends DeviceBufferLayers with FloatLayers {

  trait FloatDeviceBufferLayerApi extends DeviceBufferLayerApi {
    type Element = Float
  }

  type FloatDeviceBufferLayer <: DeviceBufferLayer with FloatDeviceBufferLayerApi

  @inject
  protected val floatDeviceBufferLayerFactory: Factory[FloatDeviceBufferLayer]

  @inject
  protected val floatDeviceBufferLayerPartialApplyRawForward: PartialApply[floatDeviceBufferLayerFactory.Constructor,
                                                                           shapeless.Witness.`"rawForward"`.T]

  @inject
  protected def floatDeviceBufferLayerRawForwardParameter
    : Do[Tape[DeviceBuffer[Float], DeviceBuffer[Float]]] <:< floatDeviceBufferLayerPartialApplyRawForward.Parameter

  trait ImplicitsApi extends super[DeviceBufferLayers].ImplicitsApi with super[FloatLayers].ImplicitsApi {

    implicit def toFloatDeviceBufferLayer[Out <: FloatDeviceBufferLayer](
        implicit implicitApply: ImplicitApply.Aux[floatDeviceBufferLayerPartialApplyRawForward.Rest, Out])
      : Layers.ToLayer.Aux[DeviceBuffer[Float], DeviceBuffer[Float], FloatDeviceBufferLayer] =
      new Layers.ToLayer[DeviceBuffer[Float], DeviceBuffer[Float]] {
        type OutputLayer = FloatDeviceBufferLayer
        def toLayer(forward: Do[Tape[DeviceBuffer[Float], DeviceBuffer[Float]]]): FloatDeviceBufferLayer = {

          implicitApply(
            floatDeviceBufferLayerPartialApplyRawForward(floatDeviceBufferLayerFactory.newInstance,
                                                         floatDeviceBufferLayerRawForwardParameter(forward)))
        }
      }
  }

  override type Implicits <: ImplicitsApi

}
