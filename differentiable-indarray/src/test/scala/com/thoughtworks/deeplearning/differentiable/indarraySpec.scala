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

  "exp(INDArray)" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 10).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      exp(weight) + 0.0
    }

    def trainMyNetwork(inputData: INDArray): Task[INDArray] = {
      train(myNetwork(inputData))
    }
    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.future.Do.doMonadErrorInstances

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
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 10).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      log(weight) + 0.0
    }

    def trainMyNetwork(inputData: INDArray): Task[INDArray] = {
      train(myNetwork(inputData))
    }
    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.future.Do.doMonadErrorInstances

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
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 10).toWeight

    def myNetwork(input: INDArray): Do[INDArrayTape] = {
      abs(weight) + 0.0
    }

    def trainMyNetwork(inputData: INDArray): Task[INDArray] = {
      train(myNetwork(inputData))
    }
    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.future.Do.doMonadErrorInstances

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
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = (Nd4j.ones(4, 4) * 10).toWeight

    def myNetwork(input: Do[INDArrayTape]): Do[INDArrayTape] = {
      dot(input, weight)
    }

    def trainMyNetwork(inputData: Do[INDArrayTape]): Task[INDArray] = {
      train(myNetwork(inputData))
    }
    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.future.Do.doMonadErrorInstances

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
    import com.thoughtworks.raii.future.Do.doMonadErrorInstances

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
    import com.thoughtworks.raii.future.Do.doMonadErrorInstances

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
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = (1 to 54).toNDArray.reshape(2, 3, 3, 3).toWeight

    def myNetwork(dimensions: Int*): Do[INDArrayTape] = {
      reshape(weight, dimensions: _*)
    }

    def trainMyNetwork(dimensions: Int*): Task[INDArray] = {
      train(myNetwork(dimensions: _*))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.future.Do.doMonadErrorInstances

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
    import com.thoughtworks.raii.future.Do.doMonadErrorInstances

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
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = (1 to 54).toNDArray.reshape(2, 3, 9).toWeight

    def myNetwork(dimensions: Int*): Do[INDArrayTape] = {
      permute(weight, dimensions: _*)
    }

    def trainMyNetwork(dimensions: Int*): Task[INDArray] = {
      train(myNetwork(dimensions: _*))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.future.Do.doMonadErrorInstances

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
    import com.thoughtworks.raii.future.Do.doMonadErrorInstances

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

  "shape(INDArray)" in {

    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Do[INDArrayTape] = Nd4j.zeros(2, 3, 3, 3).toWeight
    val task: Task[Array[Int]] = Do.run(shape(weight))

    val promise = Promise[Assertion]

    task.unsafePerformAsync { either: \/[Throwable, Array[Int]] =>
      promise.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(weightShape) => {
            weightShape(0) should be(2)
            weightShape(1) should be(3)
            weightShape(2) should be(3)
            weightShape(3) should be(3)
          }
        }
      }
    }
    promise.future
  }

  "conv2d(INDArray, INDArray, INDArray, kernel, stride, padding)" in {

    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 0.01
    }

    val input = (1 to 16).toNDArray.reshape(1, 1, 4, 4).toWeight

    val weight: Do[INDArrayTape] = Nd4j.ones(1, 1, 3, 3).toWeight
    val bias = Nd4j.zeros(1).toWeight

    def convolution(input: Do[_ <: INDArrayTape]): Do[INDArrayTape] = {
      conv2d(input, weight, bias, (3, 3), (1, 1), (1, 1))
    }

    def trainConvlountion(input: Do[_ <: INDArrayTape]): Task[INDArray] = {
      train(convolution(input))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.future.Do.doMonadErrorInstances

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

}
