package com.thoughtworks.deeplearning.double.optimizers

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait L2Regularization extends LearningRate {
  protected def l2Regularization: Double

  override def updateDouble(oldValue: Double, delta: Double): Double = {
    super.updateDouble(oldValue, delta) - l2Regularization * oldValue * currentLearningRate()
  }

}
