package com.thoughtworks.deeplearning.plugins

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
  *          import scalaz.std.vector._
  *          import scalaz.concurrent.Task
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
  *          @monadic[Task]
  *          def train: Task[Vector[Double]] = {
  *            for (iteration <- (0 until numberOfIterations).toVector) yield {
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
