package com.thoughtworks.deeplearning.plugins

/**
  * @author 杨博 (Yang Bo)
  */
trait Builtins
    extends ImplicitsSingleton
    with Layers
    with Weights
    with Logging
    with Operators
    with FloatTraining
    with FloatLiterals
    with FloatWeights
    with RawFloatLayers
    with FloatLayers
    with DoubleTraining
    with DoubleLiterals
    with DoubleWeights
    with RawDoubleLayers
    with DoubleLayers
    with INDArrayTraining
    with INDArrayLiterals
    with INDArrayWeights
    with RawINDArrayLayers
    with INDArrayLayers {

  trait ImplicitsApi
      extends super[Layers].ImplicitsApi
      with super[Weights].ImplicitsApi
      with super[Operators].ImplicitsApi
      with super[FloatTraining].ImplicitsApi
      with super[FloatLiterals].ImplicitsApi
      with super[RawFloatLayers].ImplicitsApi
      with super[DoubleTraining].ImplicitsApi
      with super[DoubleLiterals].ImplicitsApi
      with super[RawDoubleLayers].ImplicitsApi
      with super[INDArrayTraining].ImplicitsApi
      with super[INDArrayLiterals].ImplicitsApi
      with super[RawINDArrayLayers].ImplicitsApi

  type Implicits <: ImplicitsApi
}
