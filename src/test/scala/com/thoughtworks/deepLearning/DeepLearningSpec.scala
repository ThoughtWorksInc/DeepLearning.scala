package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.Differentiable.{SymbolicInput}
import org.scalatest.{FreeSpec, Matchers}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DeepLearningSpec extends FreeSpec with Matchers {

  implicit def learningRate = new Differentiable.LearningRate {
    def apply() = 0.03
  }

  "XOR" in {
    def predictXor(dsl: Dsl)(in: dsl.Array2D) = {
      import dsl._
      val deepLearning = DeepLearning(dsl)
      import deepLearning._
      sigmoid(fullyConnectedThenRelu(
        (0 until 10).foldLeft(fullyConnectedThenRelu(in, 2, 50)) { (hiddenLayer, _) =>
          fullyConnectedThenRelu(hiddenLayer, 50, 50)
        },
        50,
        2
      ))
    }

    val symbolicInput = shapeless.the[SymbolicInput {type InputSymbol[D <: Dsl] = D#Array2D}]
    val predictor = predictXor(symbolicInput.dsl)(symbolicInput.inputSymbol)
  }

}
