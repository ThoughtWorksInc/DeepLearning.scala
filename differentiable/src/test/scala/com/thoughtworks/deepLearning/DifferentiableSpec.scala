package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.Differentiable.LearningRate
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.scalatest._
import resource._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DifferentiableSpec extends FreeSpec with Matchers {

  implicit val learningRate = new LearningRate {
    override def apply(): Double = 0.0003
  }

  "Array2D dot Array2D" in {

    val network = {
      Differentiable.Dot(Differentiable.Array2DWeight(Array(Array(0.0, 5.0))),
                         Differentiable.Id[Eval[INDArray], Eval[Option[INDArray]]])
    }

    val inputBatch = Differentiable.Literal(
      Eval.now(Array(Array(2.5, -3.2, -19.5), Array(7.5, -5.4, 4.5)).toNDArray)
    )

    def train() = managed(network.forward(inputBatch)).acquireAndGet { outputBatch =>
      val loss = outputBatch.value.map(_.sumT)
      outputBatch.backward(outputBatch.value.map(Some(_)))
      loss
    }

    train().value should be(33.0)

    for (_ <- 0 until 100) {
      train().value
    }

    train().value should be < 1.0

  }

}
