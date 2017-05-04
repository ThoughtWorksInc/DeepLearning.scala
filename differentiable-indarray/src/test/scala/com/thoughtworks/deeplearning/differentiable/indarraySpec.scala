package com.thoughtworks.deeplearning.differentiable

import com.thoughtworks.deeplearning.PolyFunctions._
import com.thoughtworks.deeplearning.Tape
import com.thoughtworks.deeplearning.Tape.{Aux, Literal}
import com.thoughtworks.deeplearning.TapeTask.{predict, train}
import com.thoughtworks.deeplearning.differentiable.indarray.{INDArrayTape, Optimizer}
import com.thoughtworks.deeplearning.differentiable.indarray.Optimizer.LearningRate
import com.thoughtworks.deeplearning.differentiable.indarray.implicits._
import com.thoughtworks.each.Monadic._
import com.thoughtworks.raii.future.Do
import com.thoughtworks.raii.future.Do._
import com.thoughtworks.deeplearning.differentiable.double.DoubleTape
import com.thoughtworks.deeplearning.differentiable.double.implicits._
import com.thoughtworks.deeplearning.differentiable.double._
import com.thoughtworks.raii.transformers
import com.thoughtworks.raii.transformers.{ResourceFactoryT, ResourceT}
import com.thoughtworks.tryt.{TryT, TryTExtractor}
import org.scalactic.ErrorMessage
import org.scalatest._
import com.thoughtworks.raii.ownership.Borrowing
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.{IsMax, Sqrt}
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.ops.transforms.Transforms.{sign, sqrt}
import org.nd4j.linalg.util.ArrayUtil
import org.nd4s.Implicits._

import scalaz.concurrent.Future.futureInstance
import scala.concurrent.{ExecutionContext, Promise}
import scala.util.Try
import scalaz.concurrent.{Future, Task}
import scalaz.std.option._
import scalaz.{-\/, EitherT, MonadError, \/, \/-}
import scalaz.syntax.all._
import scalaz.std.`try`.toDisjunction
import scalaz.std.iterable._

object indarraySpec {
  implicit final class WeightData(weight: Do[INDArrayTape]) {
    def data: INDArray = {
      val task: Task[INDArrayTape] = Do.run(weight)
      val bTape: INDArrayTape = task.unsafePerformSync
      bTape.data.asInstanceOf[INDArray]
    }
  }
}

final class indarraySpec extends AsyncFreeSpec with Matchers with Inside {
  import indarraySpec._

  def trainAndAssertLossAndWeight(myNetwork: INDArray => Do[INDArrayTape],
                                  weight: Do[INDArrayTape],
                                  trainTimes: Int = 2,
                                  expectedLoss: Int = 0,
                                  expectedWeightSum: Int = -16): scala.concurrent.Future[Assertion] = {
    def trainMyNetwork(inputData: INDArray): Task[INDArray] = {
      train(myNetwork(inputData))
    }
    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.future.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to trainTimes) {
        trainMyNetwork(Nd4j.ones(4, 4)).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(Nd4j.ones(4, 4))).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be(expectedLoss)
            weight.data.sumT should be(expectedWeightSum)
        }
      }
    }
    p.future
  }

  "INDArray + INDArray" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = Nd4j.ones(4, 4).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight + input
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "INDArray + Double" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = Nd4j.ones(4, 4).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight + 1.0
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "Double + INDArray" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = Nd4j.ones(4, 4).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      1.0 + weight
    }
    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "INDArray - INDArray" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = Nd4j.ones(4, 4).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight - (-input)
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "INDArray - Double" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = Nd4j.ones(4, 4).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight - (-1.0)
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "Double - INDArray" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = (-Nd4j.ones(4, 4)).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      1.0 - weight
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 16)
  }

  "INDArray * INDArray" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 2).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight * input
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "INDArray * Double" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 2).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight * 1.0
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "Double * INDArray" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 2).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      1.0 * weight
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "INDArray / INDArray" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 2).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight / input
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "INDArray / Double" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 2).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight / 1.0
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "Double / INDArray" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = Nd4j.ones(4, 4).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      1.0 / weight
    }

    def trainMyNetwork(inputData: INDArray): Task[INDArray] = {
      train(myNetwork(inputData))
    }
    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.future.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 20000) {
        trainMyNetwork(Nd4j.ones(4, 4)).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(Nd4j.ones(4, 4))).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be < 0.5
            weight.data.sumT should be > 600.0
        }
      }
    }
    p.future
  }

  "max(INDArray,Double)" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 10).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      max(weight, 0.0)
    }

    trainAndAssertLossAndWeight(myNetwork, weight, trainTimes = 10, expectedWeightSum = 0)

  }

  "min(INDArray,Double)" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 10).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      min(weight, 100.0)
    }

    trainAndAssertLossAndWeight(myNetwork, weight, trainTimes = 10, expectedWeightSum = 0)
  }

}
