package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.any.ast.Identity
//import com.thoughtworks.deepLearning.DifferentiableAny.Ast
import com.thoughtworks.deepLearning.array2D.ast.Weight
import com.thoughtworks.deepLearning.array2D._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import any._
import org.scalatest._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DifferentiableSpec extends FreeSpec with Matchers {

  implicit val learningRate = new LearningRate {
    override def apply() = 0.0003
  }

  "Array2D dot Array2D" in {

    val network = -weight(Array(Array(0.0, 5.0))).dot(input[Array2D])

    val inputBatch = array2D.literal(Array(Array(2.5, -3.2, -19.5), Array(7.5, -5.4, 4.5)))

    def train() = {
      val outputBatch = network.forward(inputBatch)
      try {
        val loss = outputBatch.value.map(_.sumT)
        outputBatch.backward(outputBatch.value)
        loss
      } finally {
        outputBatch.close()
      }
    }

    train().value should be(-33.0)

    for (_ <- 0 until 100) {
      train().value
    }

    math.abs(train().value) should be < 1.0

  }

}
