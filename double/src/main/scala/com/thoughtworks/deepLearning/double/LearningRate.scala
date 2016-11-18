package com.thoughtworks.deepLearning.double

// TODO: Move to double library
/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait LearningRate {
  def apply(): scala.Double
}
