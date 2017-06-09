package com.thoughtworks.deeplearning
package plugins

/** A DeepLearning.scala plugin that enable [[DeepLearning.Ops.train train]] method for neural networks whose loss is a [[scala.Float]].
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait FloatTraining extends Training {
  trait ImplicitsApi extends super.ImplicitsApi with spire.std.FloatInstances
  type Implicits <: ImplicitsApi
}
