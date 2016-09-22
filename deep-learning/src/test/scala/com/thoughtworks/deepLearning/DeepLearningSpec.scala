package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.Differentiable.{SymbolicDsl, SymbolicInput}
import org.scalatest.{FreeSpec, Matchers}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DeepLearningSpec extends FreeSpec with Matchers {

  implicit def learningRate = new Differentiable.LearningRate {
    def apply() = 0.03
  }

  "XOR" in {
    def predictXor(dsl: Dsl)(input: dsl.Array2D):dsl.Array2D = {
      import dsl._
      val deepLearning = DeepLearning(dsl)
      import deepLearning._
      sigmoid(fullyConnectedThenRelu(
        (0 until 10).foldLeft(fullyConnectedThenRelu(input, 2, 50)) { (hiddenLayer, _) =>
          fullyConnectedThenRelu(hiddenLayer, 50, 50)
        },
        50,
        2
      ))
    }

    val predictorInput = shapeless.the[SymbolicInput {type Ast[D <: SymbolicDsl] = D#Array2D}]
    val predictor = predictXor(predictorInput.dsl)(predictorInput.ast)

    type LossInput[D <: Dsl] = D# ::[(D#Array2D), (D# ::)[(D#Array2D), (D#HNil)]]
    def loss(dsl: Dsl)(scoresAndLabels: LossInput[dsl.type]) = {
      val scores = scoresAndLabels.head
      val labels = scoresAndLabels.tail.head
    }

    // Does not compile because Scala compile is not able to search the implicit value with very complex dependent type
    //    val lossInput = shapeless.the[SymbolicInput {type Ast[D <: SymbolicDsl] = LossInput[D]}]
    val lossInput = SymbolicInput.hconsInput(learningRate, SymbolicInput.array2DInput, SymbolicInput.hconsInput(learningRate, SymbolicInput.array2DInput, SymbolicInput.hnilInput))
    val lossNetwork = loss(lossInput.dsl)(lossInput.ast)
  }

}
