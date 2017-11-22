package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.plugins._

trait FloatDeviceBufferLayers extends DeviceBufferLayers with FloatLayers {

  trait FloatDeviceBufferLayerApi extends DeviceBufferLayerApi {
    type Element = Float
  }

  type FloatDeviceBufferLayer <: DeviceBufferLayer with FloatDeviceBufferLayerApi

  trait ImplicitsApi extends super[DeviceBufferLayers].ImplicitsApi with super[FloatLayers].ImplicitsApi {

//    implicit def floatDeviceBufferLayerDeepLearning
//      : DeepLearning.Aux[FloatDeviceBufferLayer, DeviceBuffer[Float], DeviceBuffer[Float]] = ???
    implicit def floatDeviceBufferLayerFactory
      : Layers.ToLayer.Aux[DeviceBuffer[Float], DeviceBuffer[Float], FloatDeviceBufferLayer] = ???
  }

  override type Implicits <: ImplicitsApi

}
