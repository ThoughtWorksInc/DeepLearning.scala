package com.thoughtworks.deeplearning
package plugins

/** A DeepLearning.scala plugin that enables [[DeepLearning.Ops.train train]] method for neural networks whose loss is a [[scala.Float]].
  *
  * @author 杨博 (Yang Bo)
  */
trait FloatTraining extends Training {
  trait ImplicitsApi extends super.ImplicitsApi with algebra.instances.FloatInstances
  type Implicits <: ImplicitsApi
}
