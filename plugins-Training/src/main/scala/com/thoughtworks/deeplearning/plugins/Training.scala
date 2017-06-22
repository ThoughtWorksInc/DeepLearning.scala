package com.thoughtworks.deeplearning
package plugins

/** A DeepLearning.scala plugin that enables methods defined in [[DeepLearning.Ops]] for neural networks.
  *
  * @author 杨博 (Yang Bo)
  */
trait Training {

  trait ImplicitsApi extends DeepLearning.ToDeepLearningOps

  /** @template */
  type Implicits <: ImplicitsApi

}
