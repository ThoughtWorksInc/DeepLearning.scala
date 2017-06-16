package com.thoughtworks.deeplearning.plugins

/** A plugin that enables all other DeepLearning.scala built-in plugins.
  *
  * @example When creating a [[Builtins]] from [[com.thoughtworks.feature.Factory]],
  *
  *          {{{
  *          import com.thoughtworks.feature.Factory
  *          import com.thoughtworks.deeplearning.plugins.Builtins
  *          val hyperparameters = Factory[Builtins].newInstance()
  *          }}}
  *
  *          and `import` anything in [[implicits]],
  *
  *          {{{
  *          import hyperparameters.implicits._
  *          }}}
  *
  *          then all DeepLearning.scala built-in features should be enabled.
  *
  *          <hr/>
  *
  *          Creating weights:
  *
  *          {{{
  *          import org.nd4j.linalg.factory.Nd4j
  *          val weight: hyperparameters.INDArrayWeight = hyperparameters.INDArrayWeight(Nd4j.rand(8, 1))
  *          }}}
  *
  *          Creating neural network layers:
  *
  *          {{{
  *          import org.nd4j.linalg.api.ndarray.INDArray
  *          def fullyConnectedLayer(input: INDArray): hyperparameters.INDArrayLayer = {
  *            input dot weight
  *          }
  *          }}}
  *
  *          or loss functions:
  *
  *          {{{
  *          def hingeLoss(scores: hyperparameters.INDArrayLayer, label: INDArray): hyperparameters.DoubleLayers = {
  *            hyperparameters.max(0.0, 1.0 - label * scores).sum
  *          }
  *          }}}
  *
  *          Training:
  *          {{{
  *          val input = Nd4j.rand(4, 8)
  *          val label = Nd4j.rand(4, 1)
  *          for {
  *            lossBeforeTraining <- hingeLoss(fullyConnectedLayer(input), label).train
  *            lossAfterTraining <- hingeLoss(fullyConnectedLayer(input), label).predict
  *          } yield {
  *            lossAfterTraining should be <= lossBeforeTraining
  *          }
  *          }}}
  *
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
