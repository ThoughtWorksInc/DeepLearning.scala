package com.thoughtworks.deeplearning.seq2D.utilities

import cats._
import cats.implicits._
import org.nd4s.Implicits._
import com.thoughtworks.deeplearning.Batch._
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.Batch
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
private[deeplearning] trait Seq2DBatch extends Batch {
  override type Data = Eval[Seq[Seq[scala.Double]]]
  override type Delta = Eval[(scala.Int, scala.Int, scala.Double)]
}
