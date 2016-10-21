package com.thoughtworks.deepLearning

import org.scalatest._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DifferentiableSpec extends FreeSpec with Matchers with Inside {

  "create neuron network by macros with no weight" in {
    //    @differenitable
    //    val f: DifferentiableFunction.Aux[Eval[scala.Double], Eval[scala.Double], Eval[scala.Double], Eval[scala.Double]] = { input: Double =>
    //      input + 3.0
    //    }
    val nn = DifferentiableFunction.Add(Differentiable.Double.id, Differentiable.Double.literal(3.0))

    def train(inputValue: scala.Double) = {
      val outputBatch = nn.forward(Differentiable.Double.literal(inputValue))
      val value = outputBatch.value
      outputBatch.backward(value)
      value.value
    }
    train(5.0) should be(8.0)

    for (_ <- 0 until 100) {
      train(5.0)
    }
    train(5.0) should be(8.0)

  }

  "create neuron network by macros with a weight" in {
    //    @differenitable
    //    val f: DifferentiableFunction.Aux[Eval[scala.Double], Eval[scala.Double], Eval[scala.Double], Eval[scala.Double]] = { input: Double =>
    //      input + 3.0
    //    }
    val nn = DifferentiableFunction.Add(Differentiable.Double.id, Differentiable.Double.weight(3.0))

    def train(inputValue: scala.Double) = {
      val outputBatch = nn.forward(Differentiable.Double.literal(inputValue))
      val loss = outputBatch.value
      outputBatch.backward(loss)
      loss.value
    }
    train(5.0) should be(8.0)
    train(5.0) should be < 8.0

    for (_ <- 0 until 100) {
      train(5.0)
    }
    train(5.0) should be < 1.0

  }

}
