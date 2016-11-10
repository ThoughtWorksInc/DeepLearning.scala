package com.thoughtworks.deepLearning.array2D

import cats._
import com.thoughtworks.deepLearning.Differentiable
import com.thoughtworks.deepLearning.Differentiable._
import org.nd4j.linalg.api.ndarray.INDArray

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object utilities {

  private[array2D] type Array2D = com.thoughtworks.deepLearning.DifferentiableType[Eval[INDArray], Eval[INDArray]]

}
