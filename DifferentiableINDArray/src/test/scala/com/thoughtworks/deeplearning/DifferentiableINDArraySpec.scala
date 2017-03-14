package com.thoughtworks.deeplearning

import cats.Eval
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.DifferentiableINDArray._
import com.thoughtworks.deeplearning.DifferentiableINDArray.Optimizers.LearningRate
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.Poly._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.scalatest._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DifferentiableINDArraySpec extends FreeSpec with Matchers {

  implicit val learningRate = new LearningRate {
    override def currentLearningRate() = 0.003
  }

  "4D INDArray * 4D INDArray" in {

    def makeNetwork(implicit x: shapeless.the.`From[INDArray]`.Out) = {
      val weight = (0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4).toWeight

      weight * x
    }

    val network = makeNetwork

    val inputData = (0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4)

    def train() = {
      val outputTape = network.forward(inputData.toTape)
      try {
        val loss = outputTape.value.meanNumber.doubleValue
        outputTape.backward(outputTape.value)
        loss
      } finally {
        outputTape.close()
      }
    }

    train() should be(180.16666666666666667 +- 0.1)

    for (_ <- 0 until 100) {
      train()
    }

    math.abs(train()) should be < 1.0

  }

  // Failed due to nd4j bugs in broadcasting. TODO: Try to upgrade nd4j to a new version.
  "4D INDArray * 4D INDArray with broadcast" ignore {

    def makeNetwork(implicit x: shapeless.the.`From[INDArray]`.Out) = {
      val weight = (0 until (1 * 3 * 1 * 5)).toNDArray.reshape(1, 3, 1, 5).toWeight

      weight * x
    }

    val network = makeNetwork

    val inputData = (0 until (2 * 1 * 4 * 5)).toNDArray.reshape(2, 1, 4, 5)

    def train() = {
      val outputTape = network.forward(inputData.toTape)
      try {
        val loss = (outputTape.value: INDArray).sumT
        outputTape.backward(outputTape.value)
        loss
      } finally {
        outputTape.close()
      }
    }

    train().value should be(??? : Double)

    for (_ <- 0 until 100) {
      train().value
    }

    math.abs(train().value) should be < 1.0

  }
}
