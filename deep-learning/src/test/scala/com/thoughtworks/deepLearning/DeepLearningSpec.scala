package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.Differentiable._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.scalatest.{FreeSpec, Matchers}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DeepLearningSpec extends FreeSpec with Matchers {

  implicit def learningRate = new Differentiable.LearningRate {
    def apply() = 0.03
  }

  type Array2DPair[D <: Dsl] = D# ::[(D#Array2D), (D# ::)[(D#Array2D), (D#HNil)]]

  // Does not compile because Scala compile is not able to search the implicit value with very complex dependent type
  // def array2DPairInput = shapeless.the[SymbolicInput {type Ast[D <: SymbolicDsl] = Array2DPair[D]}]
  def array2DPairInput = SymbolicInput.hconsInput(learningRate, SymbolicInput.array2DInput, SymbolicInput.hconsInput(learningRate, SymbolicInput.array2DInput, SymbolicInput.hnilInput))

  "XOR" in {
    def predictDsl(dsl: SymbolicDsl)(input: dsl.Array2D): dsl.Array2D = {
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

    val predictInputSymbol = shapeless.the[SymbolicInput {type Ast[D <: SymbolicDsl] = D#Array2D}]
    val predictNetwork = predictDsl(predictInputSymbol.dsl)(predictInputSymbol.ast).underlying

    def lossDsl(dsl: SymbolicDsl)(scoresAndLabels: Array2DPair[dsl.type]) = {
      import dsl._
      val likelihood = scoresAndLabels.head
      val expectedLabels = scoresAndLabels.tail.head
      -(log(likelihood) * expectedLabels).reduceSum
    }

    val lossInputSymbol = array2DPairInput
    val lossNetwork = lossDsl(lossInputSymbol.dsl)(lossInputSymbol.ast).underlying

    def trainingDsl(dsl: SymbolicDsl)(inputAndLabels: Array2DPair[dsl.type]) = {
      import dsl._

      val input = inputAndLabels.head
      val expectedLabels = inputAndLabels.tail.head
      val likelihood = RichDifferentiable(predictNetwork).apply(input)
      RichDifferentiable(lossNetwork)(::(Array2D, ::(Array2D, HNil)), Double)((likelihood :: (expectedLabels :: (HNil: HNil)) (Array2D, HNil)) (dsl.Array2D, ::(Array2D, HNil)))
    }

    val trainingInputSymbol = array2DPairInput
    val trainingNetwork = trainingDsl(trainingInputSymbol.dsl)(trainingInputSymbol.ast).underlying


    val inputData = Eval.now(
      Array(
        Array(0.0, 0.0),
        Array(0.0, 1.0),
        Array(1.0, 0.0),
        Array(1.0, 1.0)
      ).toNDArray
    )

    val labelData = Eval.now(
      Array(
        Array(1.0, 0.0),
        Array(0.0, 1.0),
        Array(0.0, 1.0),
        Array(1.0, 0.0)
      ).toNDArray
    )

    val inputAndLabel = inputData :: labelData :: shapeless.HNil

    val loss = trainingNetwork.forward(Literal(inputAndLabel))
    loss.backward(loss.value)


  }

}
