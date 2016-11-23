package com.thoughtworks.deeplearning.array2D.optimizers

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait L1Regularization extends LearningRate {
  protected def l1Regularization: scala.Double

  override def updateNDArray(oldValue: INDArray, delta: INDArray): INDArray = {
    super.updateNDArray(oldValue, delta) - l1Regularization * currentLearningRate()
  }

}
