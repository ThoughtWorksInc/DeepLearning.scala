package com.thoughtworks.deepLearning.double.optimizers

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait LearningRate extends Optimizer {

  protected def currentLearningRate(): scala.Double

  override def updateDouble(oldValue: Double, delta: Double): Double = {
    oldValue - delta * currentLearningRate()
  }
}
