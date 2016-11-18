package com.thoughtworks.deepLearning.seq2D.utilities

import cats._
import cats.implicits._
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning.Batch._
import com.thoughtworks.deepLearning.Layer._
import com.thoughtworks.deepLearning.Batch
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
private[deepLearning] trait Seq2DBatch extends Batch {
  override type Data = Eval[Seq[Seq[scala.Double]]]
  override type Delta = Eval[(scala.Int, scala.Int, scala.Double)]
}
