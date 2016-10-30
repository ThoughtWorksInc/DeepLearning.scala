package com.thoughtworks.deepLearning.dsl

import cats.Eval
import org.nd4j.linalg.api.ndarray.INDArray

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait Array2D extends Any {
  override type Data = Eval[INDArray]
  override type Delta = Eval[INDArray]
}
