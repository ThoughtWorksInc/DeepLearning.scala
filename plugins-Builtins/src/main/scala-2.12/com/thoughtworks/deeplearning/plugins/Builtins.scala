package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.feature.mixins.ImplicitsSingleton

/** A plugin that enables all other DeepLearning.scala built-in plugins.
  *
  * @author 杨博 (Yang Bo)
  */
trait Builtins
    extends ImplicitsSingleton
    with Layers
    with Weights
    with Logging
    with Names
    with Operators
    with FloatTraining
    with FloatLiterals
    with FloatWeights
    with FloatLayers
    with CumulativeFloatLayers
    with DoubleTraining
    with DoubleLiterals
    with DoubleWeights
    with DoubleLayers
    with CumulativeDoubleLayers {

  trait ImplicitsApi
      extends super[Layers].ImplicitsApi
      with super[Weights].ImplicitsApi
      with super[Operators].ImplicitsApi
      with super[FloatTraining].ImplicitsApi
      with super[FloatLiterals].ImplicitsApi
      with super[FloatLayers].ImplicitsApi
      with super[DoubleTraining].ImplicitsApi
      with super[DoubleLiterals].ImplicitsApi
      with super[DoubleLayers].ImplicitsApi

  type Implicits <: ImplicitsApi

  trait LayerApi extends super[Logging].LayerApi with super[Names].LayerApi { this: Layer =>
  }

  type Layer <: LayerApi

  trait WeightApi extends super[Logging].WeightApi with super[Names].WeightApi { this: Weight =>
  }

  type Weight <: WeightApi
}
