package com.thoughtworks.deepLearning.boolean.ast

import com.thoughtworks.deepLearning.{Batch, BatchId, NeuralNetwork}
import com.thoughtworks.deepLearning.NeuralNetwork._
import com.thoughtworks.deepLearning.boolean.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class If[Input0 <: Batch, Output0 <: Batch](condition: NeuralNetwork.Aux[Input0, Boolean#ConcreteBatch],
                                                       `then`: NeuralNetwork.Aux[Input0, Output0],
                                                       `else`: NeuralNetwork.Aux[Input0, Output0])
    extends NeuralNetwork {
  override type Input = Input0
  override type Output = Output0

  override def forward(input: BatchId.Aux[Input0]) = {
    val conditionForwardPass = condition.forward(input).open()
    try {
      if (conditionForwardPass.value.value) {
        `then`.forward(input)
      } else {
        `else`.forward(input)
      }
    } finally {
      conditionForwardPass.close()
    }
  }
}
