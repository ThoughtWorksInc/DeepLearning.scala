package com.thoughtworks.deepLearning.double
package utilities

import cats._
import cats.implicits._
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning.{Batch, Differentiable}
import com.thoughtworks.deepLearning.Differentiable._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
private[deepLearning] trait DoubleMonoidBatch extends Batch {

  override type Data = Eval[scala.Double]

  override type Delta = Eval[scala.Double]

  final def monoid: Monoid[Delta] = implicitly

}
