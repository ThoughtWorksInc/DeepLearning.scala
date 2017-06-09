package com.thoughtworks.deeplearning
package plugins

/** A DeepLearning.scala plugin that enable methods defined in [[DeepLearning.Ops]] for neural networks.
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait Training {
  trait ImplicitsApi extends DeepLearning.ToDeepLearningOps
  type Implicits <: ImplicitsApi
}
