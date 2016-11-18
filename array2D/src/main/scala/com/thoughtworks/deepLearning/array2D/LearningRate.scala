package com.thoughtworks.deepLearning.array2D

import cats._
import cats.implicits._
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
trait Optimizer {
  def updateNDArray(oldValue: INDArray, delta: INDArray): INDArray
}
trait LearningRate extends Optimizer {

  protected def learningRate(): scala.Double

  override def updateNDArray(oldValue: INDArray, delta: INDArray): INDArray = {
    oldValue - delta * learningRate()
  }
}


trait L1Regularization extends LearningRate {
  protected def l1Regularization: scala.Double

  override def updateNDArray(oldValue: INDArray, delta: INDArray): INDArray = {
    super.updateNDArray(oldValue, delta) - l1Regularization * learningRate()
  }

}

trait L2Regularization extends LearningRate {
  protected def l2Regularization: scala.Double

  override def updateNDArray(oldValue: INDArray, delta: INDArray): INDArray = {
    super.updateNDArray(oldValue, delta) - oldValue * l2Regularization * learningRate()
  }

}
