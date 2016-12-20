package com.thoughtworks.deeplearning.double.optimizers

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait LearningRate extends Optimizer {

  protected def currentLearningRate(): Double

  override def updateDouble(oldValue: Double, delta: Double): Double = {
    oldValue - delta * currentLearningRate()
  }
}
