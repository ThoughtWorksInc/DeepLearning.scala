package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.compute.OpenCL
import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.feature.ImplicitApply
import com.thoughtworks.raii.asynchronous.Do

trait DeviceBufferWeights extends Weights with OpenCL {

  trait ImplicitsApi extends super[Weights].ImplicitsApi with super[OpenCL].ImplicitsApi

  type Implicits <: ImplicitsApi

  trait DeviceBufferWeightApi extends WeightApi { this: DeviceBufferWeight =>
    type Element
    override type Data = DeviceBuffer[Element]
    override type Delta = DeviceBuffer[Element] => Do[Unit]

  }

  type DeviceBufferWeight <: DeviceBufferWeightApi with Weight



}
