package com.thoughtworks.deeplearning
package differentiable

import java.util.logging.Logger

import com.thoughtworks.deeplearning.math._
import com.thoughtworks.deeplearning.differentiable.Any.{predict, train}
import com.thoughtworks.deeplearning.differentiable.INDArray.hyperparameters
import com.thoughtworks.deeplearning.differentiable.INDArray.INDArrayTape
import com.thoughtworks.deeplearning.differentiable.INDArray.implicits._
import com.thoughtworks.each.Monadic._
import com.thoughtworks.feature.{ImplicitApply, Factory}
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
import org.nd4j.linalg.api.ndarray.{INDArray => Nd4jArray}
import org.nd4j.linalg.api.ops.impl.transforms.{IsMax, Sqrt}
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.ops.transforms.Transforms.{sign, sqrt}
import org.nd4j.linalg.util.ArrayUtil
import org.nd4s.Implicits._
import shapeless.{::, HList, Lazy}
import shapeless.PolyDefns.Case
import shapeless.ops.hlist.Selector
import shapeless.record.Record

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
    def data: Nd4jArray = {
      val task: Task[INDArrayTape] = Do.run(weight)
      val bTape: Tape[Nd4jArray, Nd4jArray] = task.unsafePerformSync
      bTape.data
    }
  }

  implicit val logger: Logger = Logger.getGlobal
}

final class INDArraySpec extends AsyncFreeSpec with Matchers with Inside {
  import INDArraySpec._

  val hyperparameters = Factory[differentiable.INDArray.hyperparameters.FixedLearningRate]
    .newInstance(fixedLearningRate = 1.0)

  def trainAndAssertLossAndWeight(myNetwork: Nd4jArray => Do[INDArrayTape],
                                  weight: hyperparameters.Weight,
                                  trainTimes: Int = 2,
                                  expectedLoss: Int = 0,
                                  expectedWeightSum: Int = -16): scala.concurrent.Future[Assertion] = {
    def trainMyNetwork(inputData: Nd4jArray): Task[Nd4jArray] = {
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

    result.unsafePerformAsync { either: \/[Throwable, Nd4jArray] =>
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
//
//  implicit def optimizer: Optimizer = new LearningRate {
//    def currentLearningRate() = 1
//  }
  "Nd4jArray + Nd4jArray" in {

    // TODO: Weight should be some kind of Aux type, not singleton type
    val weight = hyperparameters.indArrayWeight(Nd4j.ones(4, 4))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      weight.forward  + input
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "Nd4jArray + Double" in {

    val weight = hyperparameters.indArrayWeight(Nd4j.ones(4, 4))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      weight + 1.0
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "Double + Nd4jArray" in {

    val weight = hyperparameters.indArrayWeight(Nd4j.ones(4, 4))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      1.0 + weight
    }
    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "Nd4jArray - Nd4jArray" in {

    val weight = hyperparameters.indArrayWeight(Nd4j.ones(4, 4))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      weight - (-input)
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "Nd4jArray - Double" in {

    val weight = hyperparameters.indArrayWeight(Nd4j.ones(4, 4))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      weight - (-1.0)
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "Double - Nd4jArray" in {

    val weight = hyperparameters.indArrayWeight((-Nd4j.ones(4, 4)))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      1.0 - weight
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 16)
  }

  "Nd4jArray * Nd4jArray" in {

    val weight = hyperparameters.indArrayWeight((Nd4j.ones(4, 4) * 2))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      weight * input
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "Nd4jArray * Double" in {

    val weight = hyperparameters.indArrayWeight((Nd4j.ones(4, 4) * 2))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      weight * 1.0
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "Double * Nd4jArray" in {

    val weight = hyperparameters.indArrayWeight((Nd4j.ones(4, 4) * 2))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      1.0 * weight
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "Nd4jArray / Nd4jArray" in {

    val weight = hyperparameters.indArrayWeight((Nd4j.ones(4, 4) * 2))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      weight / input
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "Nd4jArray / Double" in {

    val weight = hyperparameters.indArrayWeight((Nd4j.ones(4, 4) * 2))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      weight / 1.0
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "Double / Nd4jArray" in {

    val weight = hyperparameters.indArrayWeight(Nd4j.ones(4, 4))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      1.0 / weight
    }

    def trainMyNetwork(inputData: Nd4jArray): Task[Nd4jArray] = {
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

    result.unsafePerformAsync { either: \/[Throwable, Nd4jArray] =>
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

  "max(Nd4jArray,Double)" in {

    val weight = hyperparameters.indArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      max(weight, 0.0)
    }

    trainAndAssertLossAndWeight(myNetwork, weight, trainTimes = 10, expectedWeightSum = 0)

  }

  "min(Nd4jArray,Double)" in {

    val weight = hyperparameters.indArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      min(weight, 100.0)
    }

    trainAndAssertLossAndWeight(myNetwork, weight, trainTimes = 10, expectedWeightSum = 0)
  }

  "exp(Nd4jArray)" in {

    val weight = hyperparameters.indArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      exp(weight)
    }

    def trainMyNetwork(inputData: Nd4jArray): Task[Nd4jArray] = {
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

    result.unsafePerformAsync { either: \/[Throwable, Nd4jArray] =>
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

  "log(Nd4jArray)" in {

    val weight = hyperparameters.indArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      log(weight)
    }

    def trainMyNetwork(inputData: Nd4jArray): Task[Nd4jArray] = {
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

    result.unsafePerformAsync { either: \/[Throwable, Nd4jArray] =>
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

  "abs(Nd4jArray)" in {

    val weight = hyperparameters.indArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      abs(weight)
    }

    def trainMyNetwork(inputData: Nd4jArray): Task[Nd4jArray] = {
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

    result.unsafePerformAsync { either: \/[Throwable, Nd4jArray] =>
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

  "Nd4jArray dot Nd4jArray" in {

    val weight = hyperparameters.indArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: Do[INDArrayTape]): Do[INDArrayTape] = {
      dot(input, weight)
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 10) {
        train(myNetwork(Do.now(Tape.literal(Nd4j.ones(4, 4))))).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(Do.now(Tape.literal(Nd4j.ones(4, 4))))).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Nd4jArray] =>
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

  "Nd4jArray im2col (kernel,stride,padding) --forward" in {
    val hyperparameters =
      Factory[differentiable.INDArray.hyperparameters.FixedLearningRate]
        .newInstance(fixedLearningRate = 0.03)

    val weight = hyperparameters.indArrayWeight((-(1 to 54).toNDArray).reshape(2, 3, 3, 3))

    def myNetwork(kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int)): Do[INDArrayTape] = {
      im2col(weight, kernel, stride, padding)
    }

    def trainMyNetwork(kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int)): Task[Nd4jArray] = {
      train(myNetwork(kernel, stride, padding))
    }
    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val p = Promise[Assertion]

    trainMyNetwork((3, 3), (1, 1), (1, 1)).unsafePerformAsync { either: \/[Throwable, Nd4jArray] =>
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

  "Nd4jArray im2col (kernel,stride,padding) --train" in {
    val hyperparameters =
      Factory[differentiable.INDArray.hyperparameters.FixedLearningRate]
        .newInstance(fixedLearningRate = 0.01)

    val weight = hyperparameters.indArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int)): Do[INDArrayTape] = {
      im2col(weight, kernel, stride, padding)
    }

    def trainMyNetwork(kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int)): Task[Nd4jArray] = {
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

    result.unsafePerformAsync { either: \/[Throwable, Nd4jArray] =>
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

  "Nd4jArray reshape shapes --forward" in {

    val weight = hyperparameters.indArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(dimensions: Int*): Do[INDArrayTape] = {
      reshape(weight, dimensions: _*)
    }

    def trainMyNetwork(dimensions: Int*): Task[Nd4jArray] = {
      train(myNetwork(dimensions: _*))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val p = Promise[Assertion]

    trainMyNetwork(2, 3, 3, 3).unsafePerformAsync { either: \/[Throwable, Nd4jArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(result) =>
            result.sumT should be(1431.0)
        }
      }
    }

    p.future
  }

  "Nd4jArray reshape shapes --train" in {
    val hyperparameters =
      Factory[differentiable.INDArray.hyperparameters.FixedLearningRate]
        .newInstance(fixedLearningRate = 0.01)

    val weight = hyperparameters.indArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(dimensions: Int*): Do[INDArrayTape] = {
      reshape(weight, dimensions: _*)
    }

    def trainMyNetwork(dimensions: Int*): Task[Nd4jArray] = {
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

    result.unsafePerformAsync { either: \/[Throwable, Nd4jArray] =>
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

  "Nd4jArray permute dimensions --forward" in {

    val weight = hyperparameters.indArrayWeight((1 to 54).toNDArray.reshape(2, 3, 9))

    def myNetwork(dimensions: Int*): Do[INDArrayTape] = {
      permute(weight, dimensions: _*)
    }

    def trainMyNetwork(dimensions: Int*): Task[Nd4jArray] = {
      train(myNetwork(dimensions: _*))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val p = Promise[Assertion]

    trainMyNetwork(0, 2, 1).unsafePerformAsync { either: \/[Throwable, Nd4jArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(result) =>
            result.sumT should be(1431.0)
        }
      }
    }

    p.future
  }

  "Nd4jArray permute dimensions --train" in {
    val hyperparameters =
      Factory[differentiable.INDArray.hyperparameters.FixedLearningRate]
        .newInstance(fixedLearningRate = 0.01)

    val weight = hyperparameters.indArrayWeight((1 to 54).toNDArray.reshape(2, 3, 9))

    def myNetwork(dimensions: Int*): Do[INDArrayTape] = {
      permute(weight, dimensions: _*)
    }

    def trainMyNetwork(dimensions: Int*): Task[Nd4jArray] = {
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

    result.unsafePerformAsync { either: \/[Throwable, Nd4jArray] =>
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

  "conv2d(Nd4jArray, Nd4jArray, Nd4jArray, kernel, stride, padding)" in {

    val hyperparameters =
      Factory[differentiable.INDArray.hyperparameters.FixedLearningRate]
        .newInstance(fixedLearningRate = 0.01)

    val input = hyperparameters.indArrayWeight((1 to 16).toNDArray.reshape(1, 1, 4, 4))

    val weight = hyperparameters.indArrayWeight(Nd4j.ones(1, 1, 3, 3))
    val bias = hyperparameters.indArrayWeight(Nd4j.zeros(1))

    def convolution(input: Do[INDArrayTape]): Do[INDArrayTape] = {
      conv2d(input, weight, bias, (3, 3), (1, 1), (1, 1))
    }

    def trainConvlountion(input: Do[INDArrayTape]): Task[Nd4jArray] = {
      train(convolution(input))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val p = Promise[Assertion]

    val expectResult = Array(14.00, 24.00, 30.00, 22.00, 33.00, 54.00, 63.00, 45.00, 57.00, 90.00, 99.00, 69.00, 46.00,
      72.00, 78.00, 54.00).toNDArray
      .reshape(1, 1, 4, 4)

    trainConvlountion(input.forward).unsafePerformAsync { either: \/[Throwable, Nd4jArray] =>
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

  "sumT(Nd4jArray)" in {

    val weight = hyperparameters.indArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

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

  "sum(Nd4jArray,dimensions) --2 dimensions" in {

    val weight = hyperparameters.indArrayWeight((1 to 54).toNDArray.reshape(6, 9))

    def myNetwork(dimensions: Int*): Do[INDArrayTape] = {
      sum(weight, dimensions: _*)
    }

    def trainMyNetwork(): Task[Nd4jArray] = {
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

    result.unsafePerformAsync { either: \/[Throwable, Nd4jArray] =>
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
  "sum(Nd4jArray,dimensions) --4 dimensions" ignore {

    val weight = hyperparameters.indArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(dimensions: Int*): Do[INDArrayTape] = {
      sum(weight, dimensions: _*)
    }

    def trainMyNetwork(): Task[Nd4jArray] = {
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

    result.unsafePerformAsync { either: \/[Throwable, Nd4jArray] =>
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

  "mean(Nd4jArray)" in {

    val weight = hyperparameters.indArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

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

  "4D Nd4jArray * 4D Nd4jArray -- forward" in {

    val weight = hyperparameters.indArrayWeight((0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      weight * input
    }

    def trainMyNetwork(input: Nd4jArray): Task[Nd4jArray] = {
      train(myNetwork(input))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val input = (0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4)

    val p = Promise[Assertion]

    trainMyNetwork(input).unsafePerformAsync { either: \/[Throwable, Nd4jArray] =>
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

  "4D Nd4jArray * 4D Nd4jArray -- train" in {

    val weight = hyperparameters.indArrayWeight((0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4))

    def myNetwork(input: Nd4jArray): Do[INDArrayTape] = {
      weight * input
    }

    def trainMyNetwork(input: Nd4jArray): Task[Nd4jArray] = {
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

    result.unsafePerformAsync { either: \/[Throwable, Nd4jArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss: Nd4jArray) =>
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
      "abs(??? : Do[Tape[Double, Double]])" should compile
      "abs(??? : Do[_<: Tape[Double, Double]])" should compile
      "abs(??? : Do[Tape[Double, Double]])" should compile

      "abs(Nd4j.ones(2, 3, 3, 3))" should compile
      "abs(??? : Do[INDArrayTape])" should compile
      "abs(??? : Do[_<: INDArrayTape])" should compile
      "abs(??? : Do[INDArrayTape])" should compile
      "abs(??? : Do[Tape[Nd4jArray, Nd4jArray]])" should compile
      "abs(??? : Do[_<: Tape[Nd4jArray, Nd4jArray]])" should compile
      "abs(??? : Do[Tape[Nd4jArray, Nd4jArray]])" should compile
    }

    promise.future
  }
}
