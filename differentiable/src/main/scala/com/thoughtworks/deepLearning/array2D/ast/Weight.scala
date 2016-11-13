package com.thoughtworks.deepLearning
package array2D.ast

import cats._
import com.thoughtworks.deepLearning.Batch.Aux
import com.thoughtworks.deepLearning.array2D.utilities._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Weight(var rawValue: INDArray)(implicit learningRate: LearningRate)
    extends NeuralNetwork
    with Array2DSemigroupBatch {
  override type Input = Batch
  override type Output = Batch.Aux[Data, Delta]

  override def value = Eval.now(rawValue)

  override def forward(any: Input) = this

  override def backward(delta: Delta): Unit = {
    rawValue -= delta.value * learningRate()
  }

  override def close(): Unit = {}

}
