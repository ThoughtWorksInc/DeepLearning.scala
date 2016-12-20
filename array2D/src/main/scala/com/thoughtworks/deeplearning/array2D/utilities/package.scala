package com.thoughtworks.deeplearning.array2D

import cats._
import com.thoughtworks.deeplearning.ToLayer._
import org.nd4j.linalg.api.ndarray.INDArray

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object utilities {

  private[array2D] type Array2D = BackPropagationType[Eval[INDArray], Eval[INDArray]]

}
