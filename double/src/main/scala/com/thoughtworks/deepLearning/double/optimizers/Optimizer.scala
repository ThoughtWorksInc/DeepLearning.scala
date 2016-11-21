package com.thoughtworks.deepLearning.double.optimizers

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait Optimizer {
  def updateDouble(oldValue: Double, delta: Double): Double
}