package com.thoughtworks.deepLearning.core
/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait LearningRate {
  def apply(): scala.Double
}
