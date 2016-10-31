package com.thoughtworks.deepLearning
package array2D
package utilities

import cats._
import cats.implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
private[deepLearning] trait Array2DSemigroupBatch extends Batch {

  override type Data = Array2D#Data

  override type Delta = Array2D#Delta

  protected final def semigroup = new Semigroup[Delta] {
    override def combine(x: Delta, y: Delta): Delta = x.map2(y)(_ + _)
  }

}
