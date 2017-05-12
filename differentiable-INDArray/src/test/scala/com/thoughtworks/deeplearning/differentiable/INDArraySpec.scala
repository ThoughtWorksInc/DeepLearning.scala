package com.thoughtworks.deeplearning.differentiable

import com.thoughtworks.deeplearning.math._
import com.thoughtworks.deeplearning.{math, Tape, Lift}
import com.thoughtworks.deeplearning.differentiable.Any.{predict, train}
import com.thoughtworks.deeplearning.differentiable.INDArray._
import com.thoughtworks.deeplearning.differentiable.INDArray.Optimizer
import com.thoughtworks.deeplearning.differentiable.INDArray.Optimizer.LearningRate
import com.thoughtworks.deeplearning.differentiable.INDArray.implicits._
import com.thoughtworks.each.Monadic._
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import com.thoughtworks.deeplearning.differentiable.Double.DoubleTape
import com.thoughtworks.deeplearning.differentiable.Double.implicits._
import com.thoughtworks.deeplearning.differentiable.Double._
import com.thoughtworks.raii.covariant
import com.thoughtworks.raii.covariant.{Releasable, ResourceT}
import com.thoughtworks.tryt.covariant.{TryT, TryTExtractor}
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
import shapeless.Lazy
import shapeless.PolyDefns.Case

import scalaz.concurrent.Future.futureInstance
import scala.concurrent.{ExecutionContext, Promise}
import scala.util.Try
import scalaz.concurrent.{Future, Task}
import scalaz.std.option._
import scalaz.{-\/, EitherT, MonadError, \/, \/-}
import scalaz.syntax.all._
import scalaz.std.`try`.toDisjunction
import scalaz.std.iterable._

object INDArraySpec {
  implicit final class WeightData(weight: Do[INDArrayTape]) {
    def data: INDArray = {
      val task: Task[INDArrayTape] = Do.run(weight)
      val bTape: INDArrayTape = task.unsafePerformSync
      bTape.data.asInstanceOf[INDArray]
    }
  }
}

final class INDArraySpec extends AsyncFreeSpec with Matchers with Inside {
  import INDArraySpec._

  def trainAndAssertLossAndWeight(myNetwork: INDArray => Do[INDArrayTape],
                                  weight: Do[INDArrayTape],
                                  trainTimes: Int = 2,
                                  expectedLoss: Int = 0,
                                  expectedWeightSum: Int = -16): scala.concurrent.Future[Assertion] = {
    def trainMyNetwork(inputData: INDArray): Task[INDArray] = {
      train(myNetwork(inputData))
    }
    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

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

  implicit def optimizer: Optimizer = new LearningRate {
    def currentLearningRate() = 1
  }

  "INDArray + INDArray" in {

    val weight: Do[INDArrayTape] = Nd4j.ones(4, 4).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight + input
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "INDArray + Double" in {

    val weight: Do[INDArrayTape] = Nd4j.ones(4, 4).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight + 1.0
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "Double + INDArray" in {

    val weight: Do[INDArrayTape] = Nd4j.ones(4, 4).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      1.0 + weight
    }
    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "INDArray - INDArray" in {

    val weight: Do[INDArrayTape] = Nd4j.ones(4, 4).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight - (-input)
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "INDArray - Double" in {

    val weight: Do[INDArrayTape] = Nd4j.ones(4, 4).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight - (-1.0)
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "Double - INDArray" in {

    val weight: Do[INDArrayTape] = (-Nd4j.ones(4, 4)).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      1.0 - weight
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 16)
  }

  "INDArray * INDArray" in {

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 2).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight * input
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "INDArray * Double" in {

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 2).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight * 1.0
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "Double * INDArray" in {

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 2).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      1.0 * weight
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "INDArray / INDArray" in {

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 2).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight / input
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "INDArray / Double" in {

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 2).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight / 1.0
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "Double / INDArray" in {

    val weight: Do[INDArrayTape] = Nd4j.ones(4, 4).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      1.0 / weight
    }

    def trainMyNetwork(inputData: INDArray): Task[INDArray] = {
      train(myNetwork(inputData))
    }
    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

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

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 10).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      max(weight, 0.0)
    }

    trainAndAssertLossAndWeight(myNetwork, weight, trainTimes = 10, expectedWeightSum = 0)

  }

  "min(INDArray,Double)" in {

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 10).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      min(weight, 100.0)
    }

    trainAndAssertLossAndWeight(myNetwork, weight, trainTimes = 10, expectedWeightSum = 0)
  }

  "exp(INDArray)" in {

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 10).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      exp(weight)
    }

    def trainMyNetwork(inputData: INDArray): Task[INDArray] = {
      train(myNetwork(inputData))
    }
    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 50) {
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
            loss.sumT should be < 1.0
            weight.data.sumT should be < 1.0
        }
      }
    }
    p.future
  }

  "log(INDArray)" in {

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 10).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      log(weight)
    }

    def trainMyNetwork(inputData: INDArray): Task[INDArray] = {
      train(myNetwork(inputData))
    }
    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 50) {
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
            loss.sumT should be < 10.0
            weight.data.sumT should be < 22.0
        }
      }
    }
    p.future
  }

  "abs(INDArray)" in {

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 10).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      abs(weight)
    }

    def trainMyNetwork(inputData: INDArray): Task[INDArray] = {
      train(myNetwork(inputData))
    }
    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 10) {
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
            loss.sumT should be < 1.0
            weight.data.sumT should be < 1.0
        }
      }
    }
    p.future
  }

  "INDArray dot INDArray" in {

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 10).toWeight

    def myNetwork(input: Do[INDArrayTape]): Do[INDArrayTape] = {
      dot(input, weight)
    }

    def trainMyNetwork(inputData: Do[INDArrayTape]): Task[INDArray] = {
      train(myNetwork(inputData))
    }
    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 10) {
        trainMyNetwork(Nd4j.ones(4, 4).toWeight).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(Nd4j.ones(4, 4).toWeight)).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be(-1920.0)
            weight.data.sumT should be(-480.0)
        }
      }
    }
    p.future
  }

  "INDArray im2col (kernel,stride,padding) --forward" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 0.03
    }

    val weight: Do[INDArrayTape] = (-(1 to 54).toNDArray).reshape(2, 3, 3, 3).toWeight

    def myNetwork(kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int)): Do[INDArrayTape] = {
      im2col(weight, kernel, stride, padding)
    }

    def trainMyNetwork(kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int)): Task[INDArray] = {
      train(myNetwork(kernel, stride, padding))
    }
    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val p = Promise[Assertion]

    trainMyNetwork((3, 3), (1, 1), (1, 1)).unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(result) =>
            result.sumT should be(-8085.0)
        }
      }
    }

    p.future
  }

  "INDArray im2col (kernel,stride,padding) --train" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 0.01
    }

    val weight: Do[INDArrayTape] = (1 to 54).toNDArray.reshape(2, 3, 3, 3).toWeight

    def myNetwork(kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int)): Do[INDArrayTape] = {
      im2col(weight, kernel, stride, padding)
    }

    def trainMyNetwork(kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int)): Task[INDArray] = {
      train(myNetwork(kernel, stride, padding))
    }
    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 1000) {
        trainMyNetwork((3, 3), (1, 1), (1, 1)).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork((3, 3), (1, 1), (1, 1))).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be < 1.0
        }
      }
    }
    p.future
  }

  "INDArray reshape shapes --forward" in {

    val weight: Do[INDArrayTape] = (1 to 54).toNDArray.reshape(2, 3, 3, 3).toWeight

    def myNetwork(dimensions: Int*): Do[INDArrayTape] = {
      reshape(weight, dimensions: _*)
    }

    def trainMyNetwork(dimensions: Int*): Task[INDArray] = {
      train(myNetwork(dimensions: _*))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val p = Promise[Assertion]

    trainMyNetwork(2, 3, 3, 3).unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(result) =>
            result.sumT should be(1485.0)
        }
      }
    }

    p.future
  }

  "INDArray reshape shapes --train" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 0.01
    }

    val weight: Do[INDArrayTape] = (1 to 54).toNDArray.reshape(2, 3, 3, 3).toWeight

    def myNetwork(dimensions: Int*): Do[INDArrayTape] = {
      reshape(weight, dimensions: _*)
    }

    def trainMyNetwork(dimensions: Int*): Task[INDArray] = {
      train(myNetwork(dimensions: _*))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 10000) {
        trainMyNetwork(2, 3, 3, 3).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(2, 3, 3, 3)).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be < 1.0
        }
      }
    }
    p.future
  }

  "INDArray permute dimensions --forward" in {

    val weight: Do[INDArrayTape] = (1 to 54).toNDArray.reshape(2, 3, 9).toWeight

    def myNetwork(dimensions: Int*): Do[INDArrayTape] = {
      permute(weight, dimensions: _*)
    }

    def trainMyNetwork(dimensions: Int*): Task[INDArray] = {
      train(myNetwork(dimensions: _*))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val p = Promise[Assertion]

    trainMyNetwork(0, 2, 1).unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(result) =>
            result.sumT should be(1485.0)
        }
      }
    }

    p.future
  }

  "INDArray permute dimensions --train" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 0.01
    }

    val weight: Do[INDArrayTape] = (1 to 54).toNDArray.reshape(2, 3, 9).toWeight

    def myNetwork(dimensions: Int*): Do[INDArrayTape] = {
      permute(weight, dimensions: _*)
    }

    def trainMyNetwork(dimensions: Int*): Task[INDArray] = {
      train(myNetwork(dimensions: _*))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 10000) {
        trainMyNetwork(0, 2, 1).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(0, 2, 1)).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be < 1.0
        }
      }
    }
    p.future
  }

  "conv2d(INDArray, INDArray, INDArray, kernel, stride, padding)" in {

    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 0.01
    }

    val input = (1 to 16).toNDArray.reshape(1, 1, 4, 4).toWeight

    val weight: Do[INDArrayTape] = Nd4j.ones(1, 1, 3, 3).toWeight
    val bias = Nd4j.zeros(1).toWeight

    def convolution(input: Do[ INDArrayTape]): Do[INDArrayTape] = {
      conv2d(input, weight, bias, (3, 3), (1, 1), (1, 1))
    }

    def trainConvlountion(input: Do[ INDArrayTape]): Task[INDArray] = {
      train(convolution(input))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val p = Promise[Assertion]

    val expectResult = Array(14.00, 24.00, 30.00, 22.00, 33.00, 54.00, 63.00, 45.00, 57.00, 90.00, 99.00, 69.00, 46.00,
      72.00, 78.00, 54.00).toNDArray
      .reshape(1, 1, 4, 4)

    trainConvlountion(input).unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(result) =>
            result.eq(expectResult).sumT should be(16)
        }
      }
    }

    p.future
  }

  "sumT(INDArray)" in {

    val weight: Do[INDArrayTape] = (1 to 54).toNDArray.reshape(2, 3, 3, 3).toWeight

    def myNetwork(): Do[DoubleTape] = {
      sumT(weight)
    }

    def trainMyNetwork(): Task[Double] = {
      train(myNetwork())
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 54) {
        trainMyNetwork().each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork()).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Double] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss should be < 1.0
        }
      }
    }
    p.future
  }

  "sum(INDArray,dimensions) --2 dimensions" in {

    val weight: Do[INDArrayTape] = (1 to 54).toNDArray.reshape(6, 9).toWeight

    def myNetwork(dimensions: Int*): Do[INDArrayTape] = {
      sum(weight, dimensions: _*)
    }

    def trainMyNetwork(): Task[INDArray] = {
      train(myNetwork(0))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 54) {
        trainMyNetwork().each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(0)).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be < 1.0
        }
      }
    }
    p.future
  }

  // Failed due to nd4j bugs in broadcasting. TODO: Try to upgrade nd4j to a new version.
  "sum(INDArray,dimensions) --4 dimensions" ignore {

    val weight: Do[INDArrayTape] = (1 to 54).toNDArray.reshape(2, 3, 3, 3).toWeight

    def myNetwork(dimensions: Int*): Do[INDArrayTape] = {
      sum(weight, dimensions: _*)
    }

    def trainMyNetwork(): Task[INDArray] = {
      train(myNetwork(0, 1))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 54) {
        trainMyNetwork().each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(0, 1)).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be < 1.0
        }
      }
    }
    p.future
  }

  "mean(INDArray)" in {

    val weight: Do[INDArrayTape] = (1 to 54).toNDArray.reshape(2, 3, 3, 3).toWeight

    def myNetwork(): Do[DoubleTape] = {
      mean(weight)
    }

    def trainMyNetwork(): Task[Double] = {
      train(myNetwork())
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 27) {
        trainMyNetwork().each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork()).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Double] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss should be < 1.0
        }
      }
    }
    p.future
  }

  "4D INDArray * 4D INDArray -- forward" in {

    val weight = (0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight * input
    }

    def trainMyNetwork(input: INDArray): Task[INDArray] = {
      train(myNetwork(input))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val input = (0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4)

    val p = Promise[Assertion]

    trainMyNetwork(input).unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(result) =>
            result.meanNumber.doubleValue should be(180.16666666666666667 +- 0.1)
        }
      }
    }
    p.future
  }

  "4D INDArray * 4D INDArray -- train" in {

    val weight = (0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      weight * input
    }

    def trainMyNetwork(input: INDArray): Task[INDArray] = {
      train(myNetwork(input))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val input = (0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4)

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 100) {
        trainMyNetwork(input).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(input)).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss: INDArray) =>
            loss.meanNumber.doubleValue should be < 1.0
        }
      }
    }
    p.future
  }

  "should compile" in {

    val promise = Promise[Assertion]

    promise.success {
      "abs(1.0)" should compile
      "abs(??? : Do[DoubleTape])" should compile
      "abs(??? : Do[_<: DoubleTape])" should compile
      "abs(??? : Do[DoubleTape])" should compile
      "abs(??? : Do[Borrowing[Tape[Double, Double]]])" should compile
      "abs(??? : Do[_<: Borrowing[Tape[Double, Double]]])" should compile
      "abs(??? : Do[Borrowing[Tape[Double, Double]]])" should compile

      "abs(Nd4j.ones(2, 3, 3, 3))" should compile
      "abs(??? : Do[INDArrayTape])" should compile
      "abs(??? : Do[_<: INDArrayTape])" should compile
      "abs(??? : Do[INDArrayTape])" should compile
      "abs(??? : Do[Borrowing[Tape[INDArray, INDArray]]])" should compile
      "abs(??? : Do[_<: Borrowing[Tape[INDArray, INDArray]]])" should compile
      "abs(??? : Do[Borrowing[Tape[INDArray, INDArray]]])" should compile
    }

    promise.future
  }
}
