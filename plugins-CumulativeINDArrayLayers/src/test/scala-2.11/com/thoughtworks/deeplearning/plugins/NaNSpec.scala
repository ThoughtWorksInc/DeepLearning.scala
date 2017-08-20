package com.thoughtworks.deeplearning.plugins

import org.scalatest.{FreeSpec, Inside, Matchers}
import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.deeplearning.plugins.CNNObject.{Adagrad, CNNs, LearningRate}
import com.thoughtworks.deeplearning.plugins.Utils.findMaxItemIndex
import com.thoughtworks.deeplearning.plugins._
import com.thoughtworks.each.Monadic.{monadic, _}
import com.thoughtworks.feature.{Factory, ImplicitApply}
import com.thoughtworks.raii.asynchronous._

import collection.immutable.IndexedSeq
import scala.io.Source
import scala.concurrent.ExecutionContext.Implicits.global
import scalaz.std.iterable._
import scalaz.syntax.all._
import scalaz.std.stream._
import com.thoughtworks.future._

import scala.concurrent.Await
import scala.concurrent.duration.Duration
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j.argMax
import org.nd4j.linalg.api.ndarray.INDArray
import com.thoughtworks.deeplearning.plugins.Operators._
import com.thoughtworks.deeplearning.plugins.ReadCIFARToNDArray.{TestImageAndLabels, TrainData}
import com.thoughtworks.feature.Factory
import org.joda.time.LocalTime

import scala.language.postfixOps
import scala.util.Random

import plotly.Scatter
import plotly.Plotly._
import plotly._

class NaNSpec extends FreeSpec {
  "NaN should not happen when run this test on GPU" in {
    val hyperparameters =
      Factory[
        LearningRate with Logging with ImplicitsSingleton with DoubleTraining with INDArrayTraining with INDArrayLiterals with DoubleLiterals with CumulativeDoubleLayers with Operators with CumulativeINDArrayLayers with CNNs]
        .newInstance(learningRate = 0.003)

    import hyperparameters.INDArrayWeight
    import hyperparameters.DoubleLayer
    import hyperparameters.INDArrayLayer
    import hyperparameters.implicits._

    val TrainingQuestions: INDArray = {
      import org.nd4s.Implicits._
      Array(
        Array(0, 1, 2),
        Array(4, 7, 10),
        Array(13, 15, 17)
      ).toNDArray
    }

    val ExpectedAnswers: INDArray = {
      import org.nd4s.Implicits._
      Array(
        Array(3),
        Array(13),
        Array(19)
      ).toNDArray
    }

    def initialValueOfRobotWeight: INDArray = {
      //    Nd4j.randn(3, 1)
      import org.nd4s.Implicits._
      Nd4j.ones(3, 1) * 0.01
    }

    import hyperparameters.INDArrayWeight
    val robotWeight = INDArrayWeight(initialValueOfRobotWeight)

    def iqTestRobot(questions: INDArray): INDArrayLayer = {
      0.0 + questions dot robotWeight
    }

    def squareLoss(questions: INDArray, expectAnswer: INDArray): DoubleLayer = {
      val difference = iqTestRobot(questions) - expectAnswer
      (difference * difference).mean
    }

    val TotalIterations = 100

    @monadic[Future]
    def train: Future[Stream[Double]] = {
      for (iteration <- (0 until TotalIterations).toStream) yield {
        val loss = squareLoss(TrainingQuestions, ExpectedAnswers).train.each
        println("loss:" + loss)
        loss
      }
    }

    val lossByTime: Stream[Double] =
      Await.result(train.toScalaFuture, Duration.Inf)
  }
}
