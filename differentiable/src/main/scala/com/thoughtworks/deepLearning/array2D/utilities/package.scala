package com.thoughtworks.deepLearning.array2D

import com.thoughtworks.deepLearning.Ast._
import com.thoughtworks.deepLearning.Batch._
import org.nd4j.linalg.api.ndarray.INDArray

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object utilities {

  private[deepLearning] def sumAs(outputDeltaValue: INDArray, shape: Array[Int]) = {
    shape match {
      case Array(1, 1) => outputDeltaValue.sum(0, 1)
      case Array(_, 1) => outputDeltaValue.sum(1)
      case Array(1, _) => outputDeltaValue.sum(0)
      case Array(_, _) => outputDeltaValue
    }
  }

}
