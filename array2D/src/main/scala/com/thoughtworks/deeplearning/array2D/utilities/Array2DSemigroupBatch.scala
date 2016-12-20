package com.thoughtworks.deeplearning
package array2D
package utilities

import cats._
import cats.implicits._
import com.thoughtworks.deeplearning.Layer.Batch
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
private[deeplearning] trait Array2DSemigroupBatch extends Batch {

  override type Data = Eval[INDArray]

  override type Delta = Eval[INDArray]

  protected final def semigroup = new Semigroup[Delta] {
    override def combine(x: Delta, y: Delta): Delta = x.map2(y)(_ + _)
  }

}
