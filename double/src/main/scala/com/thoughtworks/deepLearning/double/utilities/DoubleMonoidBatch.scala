package com.thoughtworks.deepLearning.double
package utilities

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.Batch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
private[deepLearning] trait DoubleMonoidBatch extends Batch {

  override type Data = Eval[scala.Double]

  override type Delta = Eval[scala.Double]

  protected final def monoid = implicitly[Monoid[Delta]]

}
