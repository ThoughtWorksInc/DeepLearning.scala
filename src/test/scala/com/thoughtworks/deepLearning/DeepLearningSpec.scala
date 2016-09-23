package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.Differentiable.{Batch, SymbolicDsl, SymbolicInput}
import org.nd4j.linalg.api.ndarray.INDArray
import org.scalatest.{FreeSpec, Matchers}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DeepLearningSpec extends FreeSpec with Matchers {

  implicit def learningRate = new Differentiable.LearningRate {
    def apply() = 0.03
  }

  "XOR" in {
    def predictXor(dsl: SymbolicDsl)(input: dsl.Array2D): dsl.Array2D = {
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
    val predictor: Differentiable.Aux[Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]], Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]] = predictXor(predictorInput.dsl)(predictorInput.ast).underlying

    type Array2DPair[D <: Dsl] = D# ::[(D#Array2D), (D# ::)[(D#Array2D), (D#HNil)]]
    def loss(dsl: SymbolicDsl)(scoresAndLabels: Array2DPair[dsl.type]) = {
      import dsl._
      val likelihood = scoresAndLabels.head
      val expectedLabels = scoresAndLabels.tail.head
      -(log(likelihood) * expectedLabels).reduceSum
    }

    // Does not compile because Scala compile is not able to search the implicit value with very complex dependent type
    // val lossInput = shapeless.the[SymbolicInput {type Ast[D <: SymbolicDsl] = Array2DPair[D]}]

    val lossInput = SymbolicInput.hconsInput(learningRate, SymbolicInput.array2DInput, SymbolicInput.hconsInput(learningRate, SymbolicInput.array2DInput, SymbolicInput.hnilInput))
    val lossNetwork = loss(lossInput.dsl)(lossInput.ast).underlying

    def train(dsl: SymbolicDsl)(inputAndLabels: Array2DPair[dsl.type]) = {
      import dsl._

      val input = inputAndLabels.head
      val expectedLabels = inputAndLabels.tail.head
      val likelihood = RichDifferentiable(predictor).apply(input)
      RichDifferentiable(lossNetwork)(::(Array2D, ::(Array2D, HNil)), Double)((likelihood :: (expectedLabels :: (HNil: HNil)) (Array2D, HNil)) (dsl.Array2D, ::(Array2D, HNil)))
    }

  }

}
