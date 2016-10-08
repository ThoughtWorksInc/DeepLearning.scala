package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.Differentiable._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.scalatest.{FreeSpec, Matchers}

import scala.util.Random

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DeepLearningSpec extends FreeSpec with Matchers {

  implicit def learningRate = new Differentiable.LearningRate {
    def apply() = 0.003
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
      val inputLayerOutput = fullyConnectedThenRelu(input, 2, 100)
      val hiddenLayerOutput = (0 until 3).foldLeft(inputLayerOutput) { (hiddenLayer, _) =>
        fullyConnectedThenRelu(hiddenLayer, 100, 100)
      }
      val scores = fullyConnectedThenRelu(hiddenLayerOutput, 100, 2)

      // softmax
      exp(scores) / exp(scores).sum(1)
    }

    val predictInputSymbol = shapeless.the[SymbolicInput {type Ast[D <: SymbolicDsl] = D#Array2D}]
    val predictNetwork:
    Differentiable.Aux[
      Batch.Aux[
        Eval[INDArray],
        Eval[Option[INDArray]]
      ],
      Batch.Aux[
        Eval[INDArray],
        Eval[Option[INDArray]]
      ]
    ] = predictDsl(predictInputSymbol.dsl)(predictInputSymbol.ast).underlying

    def lossDsl(dsl: SymbolicDsl)(scoresAndLabels: Array2DPair[dsl.type]): dsl.Double = {
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
      // val likelihood = predictNetwork(input)
      // lossNetwork(likelihood :: expectedLabels :: HNil)

      val richPredictNetwork = RichDifferentiable(predictNetwork)
      val richLossNetwork = RichDifferentiable(lossNetwork)(::(Array2D, ::(Array2D, HNil)), Double)

      val likelihood = richPredictNetwork(input)
      richLossNetwork((likelihood :: (expectedLabels :: (HNil: HNil)) (Array2D, HNil)) (dsl.Array2D, ::(Array2D, HNil)))
    }

    val trainingInputSymbol = array2DPairInput
    val trainingNetwork = trainingDsl(trainingInputSymbol.dsl)(trainingInputSymbol.ast).underlying


    //
    //    val labelData = Eval.now(
    //      Array(
    //        Array(1.0, 0.0),
    //        Array(0.0, 1.0),
    //        Array(0.0, 1.0),
    //        Array(1.0, 0.0)
    //      ).toNDArray
    //    )
    //
    //    val inputAndLabelData = inputData :: labelData :: shapeless.HNil

    val inputData = Eval.now(
      Array(
        Array(0.0, 0.0),
        Array(0.0, 1.0),
        Array(1.0, 0.0),
        Array(1.0, 1.0)
      ).toNDArray
    )
    for (_ <- 0 until 100) {
      val predictedLabel = predictNetwork.forward(Literal(inputData))
      predictedLabel.backward(Eval.now[Option[INDArray]](None))
      val predictedLabelValue = predictedLabel.value.value
      println(predictedLabelValue)
      println()
      for (_ <- 0 until 10) {
        val BatchSize = 5
        def inputAndLabelData = {

          val inputData = (for {
            _ <- 0 until BatchSize
          } yield {
            Array(Random.nextInt(2), Random.nextInt(2))
          }) (collection.breakOut(Array.canBuildFrom))

          val labelData = for {
            Array(left, right) <- inputData
          } yield {
            val result = (left == 1) ^ (right == 1)
            Array(if (result) 0 else 1, if (result) 1 else 0)
          }
          Literal(Eval.now(inputData.toNDArray) :: Eval.now(labelData.toNDArray) :: shapeless.HNil)
        }
        val loss = trainingNetwork.forward(inputAndLabelData)
        loss.backward(loss.value)
      }
    }

    val predictedLabel = predictNetwork.forward(Literal(inputData))
    predictedLabel.backward(Eval.now[Option[INDArray]](None))
    val predictedLabelValue = predictedLabel.value.value
    println(predictedLabelValue)
    predictedLabelValue(0, 0) should be > 0.5
    predictedLabelValue(0, 1) should be < 0.5
    predictedLabelValue(1, 0) should be < 0.5
    predictedLabelValue(1, 1) should be > 0.5
    predictedLabelValue(2, 0) should be < 0.5
    predictedLabelValue(2, 1) should be > 0.5
    predictedLabelValue(3, 0) should be > 0.5
    predictedLabelValue(3, 1) should be < 0.5

  }

}
