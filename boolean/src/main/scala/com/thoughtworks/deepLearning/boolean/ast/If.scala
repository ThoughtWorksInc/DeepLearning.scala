package com.thoughtworks.deepLearning.boolean.ast

import com.thoughtworks.deepLearning.{Batch, BatchId, NeuralNetwork}
import com.thoughtworks.deepLearning.NeuralNetwork._
import com.thoughtworks.deepLearning.boolean.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class If[Input0 <: Batch, OutputData0, OutputDelta0](
    condition: NeuralNetwork.Aux[Input0, Boolean#Batch],
    `then`: NeuralNetwork.Aux[Input0, Batch.Aux[OutputData0, OutputDelta0]],
    `else`: NeuralNetwork.Aux[Input0, Batch.Aux[OutputData0, OutputDelta0]])
    extends NeuralNetwork {
  override type Input = Input0
  override type Output = Batch.Aux[OutputData0, OutputDelta0]

  override def forward(input: BatchId.Aux[Input0]) = {
    val conditionId = condition.forward(input)
    new BatchId {
      type Open = Output
      override def open(): Open = {
        val conditionBatch = conditionId.open()
        val underlying = if (conditionBatch.value.value) {
          `then`.forward(input).open()
        } else {
          `else`.forward(input).open()
        }
        new Batch {
          type Data = OutputData0
          type Delta = OutputDelta0
          override def backward(delta: Delta): Unit = underlying.backward(delta)
          override def value: Data = underlying.value
          override def close(): Unit = {
            conditionBatch.close()
            underlying.close()
          }
        }
      }
    }
  }
}
