package com.thoughtworks.deeplearning.array2D.optimizers

import org.nd4j.linalg.api.ndarray.INDArray

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait Optimizer {
  def updateNDArray(oldValue: INDArray, delta: INDArray): INDArray
}
