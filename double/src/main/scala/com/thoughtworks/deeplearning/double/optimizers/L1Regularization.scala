package com.thoughtworks.deeplearning.double.optimizers

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait L1Regularization extends LearningRate {
  protected def l1Regularization: scala.Double

  override def updateDouble(oldValue: Double, delta: Double): Double = {
    super.updateDouble(oldValue, delta) - math.signum(oldValue) * l1Regularization * currentLearningRate()
  }

}