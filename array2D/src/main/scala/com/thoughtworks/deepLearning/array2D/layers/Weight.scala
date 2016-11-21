package com.thoughtworks.deepLearning
package array2D.layers

import cats._
import com.thoughtworks.deepLearning.array2D.optimizers.Optimizer
import com.thoughtworks.deepLearning.array2D.utilities._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Weight(var rawValue: INDArray)(implicit optimizer: Optimizer)
    extends Layer
    with Array2DSemigroupBatch
    with BatchId {
  override type Input = Batch
  override type Output = Batch.Aux[Data, Delta]
  override type Open = Output

  override def open() = this

  override def value = Eval.now(rawValue)

  override def forward(any: BatchId.Aux[Input]) = this

  override def backward(delta: Delta): Unit = {
    synchronized {
      rawValue = optimizer.updateNDArray(rawValue, delta.value)
    }
  }

  override def close(): Unit = {}

}
