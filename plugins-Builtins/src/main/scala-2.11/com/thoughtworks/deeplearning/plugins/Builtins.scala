package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.feature.mixins.ImplicitsSingleton

/** A plugin that enables all other DeepLearning.scala built-in plugins.
  *
  * @example When creating a [[Builtins]] from [[com.thoughtworks.feature.Factory]],
  *
  *          {{{
  *          import com.thoughtworks.feature.Factory
  *          val hyperparameters = Factory[plugins.Builtins].newInstance()
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
  *          import org.nd4j.linalg.api.ndarray.INDArray
  *          }}}
  *          {{{
  *          val numberOfInputFeatures = 8
  *          val numberOfOutputFeatures = 1
  *          val initialValueOfWeight: INDArray = Nd4j.rand(numberOfInputFeatures, numberOfOutputFeatures)
  *          val weight: hyperparameters.INDArrayWeight = hyperparameters.INDArrayWeight(initialValueOfWeight)
  *          }}}
  *
  *          Creating neural network layers,
  *
  *          {{{
  *          def fullyConnectedLayer(input: INDArray): hyperparameters.INDArrayLayer = {
  *            input dot weight
  *          }
  *          }}}
  *
  *          or loss functions:
  *
  *          {{{
  *          def hingeLoss(scores: hyperparameters.INDArrayLayer, label: INDArray): hyperparameters.DoubleLayer = {
  *            hyperparameters.max(0.0, 1.0 - label * scores).sum
  *          }
  *          }}}
  *
  *          Training:
  *          {{{
  *          import scalaz.std.stream._
  *          import com.thoughtworks.future._
  *          import com.thoughtworks.each.Monadic._
  *          }}}
  *
  *          {{{
  *          val batchSize = 4
  *          val numberOfIterations = 10
  *          val input = Nd4j.rand(batchSize, numberOfInputFeatures)
  *          val label = Nd4j.rand(batchSize, numberOfOutputFeatures)
  *          }}}
  *
  *          {{{
  *          @monadic[Future]
  *          def train: Future[Stream[Double]] = {
  *            for (iteration <- (0 until numberOfIterations).toStream) yield {
  *              hingeLoss(fullyConnectedLayer(input), label).train.each
  *            }
  *          }
  *          }}}
  *
  *          When the training is done,
  *          the loss of the last iteration should be no more than the loss of the first iteration
  *
  *          {{{
  *          train.map { lossesByIteration =>
  *            lossesByIteration.last should be <= lossesByIteration.head
  *          }
  *          }}}
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
    with CumulativeDoubleLayers
    with INDArrayTraining
    with INDArrayLiterals
    with INDArrayWeights
    with INDArrayLayers
    with CumulativeINDArrayLayers {

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
      with super[INDArrayTraining].ImplicitsApi
      with super[INDArrayLiterals].ImplicitsApi
      with super[INDArrayLayers].ImplicitsApi

  type Implicits <: ImplicitsApi

  trait LayerApi extends super[Logging].LayerApi with super[Names].LayerApi { this: Layer =>
  }

  type Layer <: LayerApi

  trait WeightApi extends super[Logging].WeightApi with super[Names].WeightApi { this: Weight =>
  }

  type Weight <: WeightApi
}
