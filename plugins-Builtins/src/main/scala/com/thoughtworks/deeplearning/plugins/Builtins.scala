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
    with FloatLiterals
    with FloatWeights
    with FloatLayers
    with CumulativeFloatLayers
    with DoubleLiterals
    with DoubleWeights
    with DoubleLayers
    with CumulativeDoubleLayers
    with TensorLiterals
    with TensorWeights
    with TensorLayers
    with CumulativeTensorLayers {

  trait ImplicitsApi
      extends super[Layers].ImplicitsApi
      with super[Weights].ImplicitsApi
      with super[Operators].ImplicitsApi
      with super[FloatLiterals].ImplicitsApi
      with super[FloatLayers].ImplicitsApi
      with super[DoubleLiterals].ImplicitsApi
      with super[DoubleLayers].ImplicitsApi
      with super[TensorLiterals].ImplicitsApi
      with super[TensorLayers].ImplicitsApi

  type Implicits <: ImplicitsApi

  trait DifferentiableApi extends super[Logging].DifferentiableApi with super[Names].DifferentiableApi
  type Differentiable <: DifferentiableApi
}
