package com.thoughtworks.deepLearning

object DeepLearning {

}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait DeepLearning {
  val dsl: Dsl

  import dsl._

  def sigmoid(input: Array2D) = {
    Double(1.0) / (Double(1.0) + exp(-input))
  }

  def relu(input: Array2D) = {
    max(input, Double(0.0))
  }

  def fullyConnected(input: Array2D, weight: Array2D, bias: Array2D) = {
    (input dot weight) + bias
  }

  def fullyConnectedThenRelu(input: Array2D, inputSize: Int, outputSize: Int) = {
    val weight = Array2D.randn(inputSize, outputSize) / Double(math.sqrt(inputSize.toDouble / 2.0))
    val bias = Array2D.zeros(outputSize)
    relu(fullyConnected(input, weight, bias))
  }

}
