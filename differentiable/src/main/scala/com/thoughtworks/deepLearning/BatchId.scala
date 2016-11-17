package com.thoughtworks.deepLearning

import cats._
import cats.implicits._
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait BatchId {
  type Open <: Batch
  def open(): Open
}

object BatchId {
  type Aux[+Open0 <: Batch] = BatchId {
    type Open <: Open0
  }
}
