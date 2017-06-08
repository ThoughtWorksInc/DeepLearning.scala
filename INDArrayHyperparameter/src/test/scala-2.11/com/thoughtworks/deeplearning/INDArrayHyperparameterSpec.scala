package com.thoughtworks.deeplearning

import java.util.logging.Logger

import Loss.train

import com.thoughtworks.deeplearning.math._
import com.thoughtworks.each.Monadic._
import com.thoughtworks.feature.{ImplicitApply, Factory}
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import com.thoughtworks.raii.covariant
import com.thoughtworks.raii.covariant.{Releasable, ResourceT}
import com.thoughtworks.tryt.covariant.{TryT, TryTExtractor}
import org.scalactic.ErrorMessage
import org.scalatest._
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
import ResourceT._

object INDArrayHyperparameterSpec {

  def predict[OutputData, OutputDelta](forward: Do[Tape[OutputData, OutputDelta]]): Task[OutputData] = {
    val Do(doOutputData) = forward.map(_.data)
    new Task(ResourceT.run(ResourceT(doOutputData).map(toDisjunction)))
  }

  implicit final class WeightData(weight: Do[Tape[INDArray, INDArray]]) {
    def data: INDArray = {
      val task: Task[Tape[INDArray, INDArray]] = Do.run(weight)
      val bTape: Tape[INDArray, INDArray] = task.unsafePerformSync
      bTape.data
    }
  }

  trait FixedLearningRate extends LearningRate {
    val fixedLearningRate: scala.Double
    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi {
      final def learningRate: scala.Double = fixedLearningRate
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi
  }

  trait LearningRate extends INDArrayHyperparameter {
    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi {
      def learningRate: scala.Double
      abstract override def delta: INDArray = super.delta * learningRate
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi
  }

  trait L1Regularization extends INDArrayHyperparameter {
    def l1Regularization: INDArray
    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi {
      abstract override def delta: INDArray = super.delta + sign(weight.data) * l1Regularization
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi
  }
  trait L2Regularization extends INDArrayHyperparameter {
    def l2Regularization: INDArray
    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi {
      abstract override def delta: INDArray = super.delta + weight.data * l2Regularization
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi
  }

  trait Momentum extends INDArrayHyperparameter {
    trait INDArrayWeightApi extends super.INDArrayWeightApi { this: INDArrayWeight =>
      def mu: scala.Double = 0.9
      var v: INDArray = Nd4j.zeros(data.shape: _*)
    }

    override type INDArrayWeight <: INDArrayWeightApi

    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi {

      private lazy val delta0: INDArray = {
        import weight._
        v = super.delta + v * mu
        v
      }
      abstract override def delta: INDArray = delta0
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi
  }

  trait NesterovMomentum extends Momentum {
    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi {
      abstract override lazy val delta: INDArray = {
        import weight._
        val vPrev = v
        vPrev * (-mu) + super.delta * (1 + mu)
      }
    }
    override type INDArrayOptimizer <: super.INDArrayOptimizerApi with INDArrayOptimizerApi
  }

  /**
    * @note This [[Adagrad]] hyperparameter is usually used before global [[LearningRate]]. e.g. `Adagrad with FixedLearningRate`, not `FixedLearningRate with Adagrad`
    */
  trait Adagrad extends INDArrayHyperparameter {
    trait INDArrayWeightApi extends super.INDArrayWeightApi { this: INDArrayWeight =>
      var cache: INDArray = Nd4j.zeros(data.shape: _*)
    }

    override type INDArrayWeight <: INDArrayWeightApi

    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi {
      def eps: scala.Double = 1e-4

      abstract override lazy val delta: INDArray = {
        import weight._
        cache = cache + super.delta * super.delta
        super.delta / (sqrt(cache) + eps)
      }
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi
  }

  /**
    * @note This [[RMSprop]] hyperparameter is usually used before global [[LearningRate]]. e.g. `RMSprop with FixedLearningRate`, not `FixedLearningRate with RMSprop`
    */
  trait RMSprop extends INDArrayHyperparameter {
    trait INDArrayWeightApi extends super.INDArrayWeightApi { this: INDArrayWeight =>
      var cache: INDArray = Nd4j.zeros(data.shape: _*)
    }

    override type INDArrayWeight <: INDArrayWeightApi

    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi {
      def eps: scala.Double = 1e-4
      def decayRate: scala.Double = 0.99
      abstract override lazy val delta: INDArray = {
        import weight._
        cache = cache * decayRate + super.delta * super.delta * (1.0 - decayRate)
        super.delta / (sqrt(cache) + eps)
      }
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi
  }

  trait Adam extends INDArrayHyperparameter {

    trait INDArrayWeightApi extends super.INDArrayWeightApi { this: INDArrayWeight =>
      var m: INDArray = Nd4j.zeros(data.shape: _*)
      var v: INDArray = Nd4j.zeros(data.shape: _*)
    }

    override type INDArrayWeight <: INDArrayWeightApi

    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi {
      def beta1: scala.Double = 0.9
      def beta2: scala.Double = 0.999
      def eps: scala.Double = 1e-8
      abstract override lazy val delta: INDArray = {
        import weight._
        m = m * beta1 + super.delta * (1.0 - beta1)
        v = v * beta2 + (super.delta * super.delta) * (1.0 - beta2)
        m / (sqrt(v) + eps)
      }
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi
  }
  implicit val logger: Logger = Logger.getGlobal
}

final class INDArrayHyperparameterSpec extends AsyncFreeSpec with Matchers with Inside {
  import INDArrayHyperparameterSpec._

  val hyperparameters = Factory[FixedLearningRate].newInstance(logger = Logger.getGlobal, fixedLearningRate = 1.0)
  import hyperparameters.implicits._
  def trainAndAssertLossAndWeight(myNetwork: INDArray => Do[Tape[INDArray, INDArray]],
                                  weight: hyperparameters.INDArrayWeight,
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
//
//  implicit def optimizer: INDArrayOptimizerApi = new LearningRate {
//    def currentLearningRate() = 1
//  }
  "INDArray + INDArray" in {

    // TODO: INDArrayWeightApi should be some kind of Aux type, not singleton type
    val weight = hyperparameters.INDArrayWeight(Nd4j.ones(4, 4))
    def l[From](from: From)(implicit lift: Lift.Aux[From, INDArray, INDArray]) = lift(from)
    l(weight)(hyperparameters.implicits.liftINDArrayWeight)

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
      weight + input
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "INDArray + Double" in {

    val weight = hyperparameters.INDArrayWeight(Nd4j.ones(4, 4))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
      weight + 1.0
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "Double + INDArray" in {

    val weight = hyperparameters.INDArrayWeight(Nd4j.ones(4, 4))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
      1.0 + weight
    }
    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "INDArray - INDArray" in {

    val weight = hyperparameters.INDArrayWeight(Nd4j.ones(4, 4))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
      weight - (-input)
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "INDArray - Double" in {

    val weight = hyperparameters.INDArrayWeight(Nd4j.ones(4, 4))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
      weight - (-1.0)
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "Double - INDArray" in {

    val weight = hyperparameters.INDArrayWeight((-Nd4j.ones(4, 4)))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
      1.0 - weight
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 16)
  }

  "INDArray * INDArray" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 2))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
      weight * input
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "INDArray * Double" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 2))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
      weight * 1.0
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "Double * INDArray" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 2))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
      1.0 * weight
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "INDArray / INDArray" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 2))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
      weight / input
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "INDArray / Double" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 2))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
      weight / 1.0
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "Double / INDArray" in {

    val weight = hyperparameters.INDArrayWeight(Nd4j.ones(4, 4))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
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
    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
      max(weight, 0.0)
    }

    trainAndAssertLossAndWeight(myNetwork, weight, trainTimes = 10, expectedWeightSum = 0)

  }

  "min(INDArray,Double)" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
      min(weight, 100.0)
    }

    trainAndAssertLossAndWeight(myNetwork, weight, trainTimes = 10, expectedWeightSum = 0)
  }

  "exp(INDArray)" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
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

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
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

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
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

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: Do[Tape[INDArray, INDArray]]): Do[Tape[INDArray, INDArray]] = {
      hyperparameters.dot(input, weight)
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 10) {
        train(myNetwork(Do.now(Tape(Nd4j.ones(4, 4), Function.const(Future.now(())))))).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(Do.now(Tape(Nd4j.ones(4, 4), Function.const(Future.now(())))))).each
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
    val hyperparameters =
      Factory[FixedLearningRate]
        .newInstance(logger = Logger.getGlobal, fixedLearningRate = 0.03)
    import hyperparameters.implicits._
    val weight = hyperparameters.INDArrayWeight((-(1 to 54).toNDArray).reshape(2, 3, 3, 3))

    def myNetwork(kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int)): Do[Tape[INDArray, INDArray]] = {
      hyperparameters.im2col(weight, kernel, stride, padding)
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
    val hyperparameters =
      Factory[FixedLearningRate]
        .newInstance(logger = Logger.getGlobal, fixedLearningRate = 0.01)
    import hyperparameters.implicits._
    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int)): Do[Tape[INDArray, INDArray]] = {
      hyperparameters.im2col(weight, kernel, stride, padding)
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

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(dimensions: Int*): Do[Tape[INDArray, INDArray]] = {
      hyperparameters.reshape(weight, dimensions: _*)
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
            result.sumT should be(1431.0)
        }
      }
    }

    p.future
  }

  "INDArray reshape shapes --train" in {
    val hyperparameters =
      Factory[FixedLearningRate]
        .newInstance(logger = Logger.getGlobal, fixedLearningRate = 0.01)
    import hyperparameters.implicits._
    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(dimensions: Int*): Do[Tape[INDArray, INDArray]] = {
      hyperparameters.reshape(weight, dimensions: _*)
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

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 9))

    def myNetwork(dimensions: Int*): Do[Tape[INDArray, INDArray]] = {
      hyperparameters.permute(weight, dimensions: _*)
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
            result.sumT should be(1431.0)
        }
      }
    }

    p.future
  }

  "INDArray permute dimensions --train" in {
    val hyperparameters =
      Factory[FixedLearningRate]
        .newInstance(logger = Logger.getGlobal, fixedLearningRate = 0.01)
    import hyperparameters.implicits._
    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 9))

    def myNetwork(dimensions: Int*): Do[Tape[INDArray, INDArray]] = {
      hyperparameters.permute(weight, dimensions: _*)
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

    val hyperparameters =
      Factory[FixedLearningRate]
        .newInstance(logger = Logger.getGlobal, fixedLearningRate = 0.01)
    import hyperparameters.implicits._
    val input = (1 to 16).toNDArray.reshape(1, 1, 4, 4)

    val weight = hyperparameters.INDArrayWeight(Nd4j.ones(1, 1, 3, 3))
    val bias = hyperparameters.INDArrayWeight(Nd4j.zeros(1))

    def convolution(input: INDArray): Do[Tape[INDArray, INDArray]] = {
      hyperparameters.conv2d(input, weight, bias, (3, 3), (1, 1), (1, 1))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val p = Promise[Assertion]

    val expectResult = Array(14.00, 24.00, 30.00, 22.00, 33.00, 54.00, 63.00, 45.00, 57.00, 90.00, 99.00, 69.00, 46.00,
      72.00, 78.00, 54.00).toNDArray
      .reshape(1, 1, 4, 4)

    train(convolution(input)).unsafePerformAsync { either: \/[Throwable, INDArray] =>
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

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(): Do[Tape[Double, Double]] = {
      hyperparameters.sumT(weight)
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

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(6, 9))

    def myNetwork(dimensions: Int*): Do[Tape[INDArray, INDArray]] = {
      hyperparameters.sum(weight, dimensions: _*)
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

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(dimensions: Int*): Do[Tape[INDArray, INDArray]] = {
      hyperparameters.sum(weight, dimensions: _*)
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

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(): Do[Tape[Double, Double]] = {
      hyperparameters.mean(weight)
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

    val weight = hyperparameters.INDArrayWeight((0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
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

    val weight = hyperparameters.INDArrayWeight((0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4))

    def myNetwork(input: INDArray): Do[Tape[INDArray, INDArray]] = {
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
      "abs(??? : Do[Tape[Double,Double]])" should compile
      "abs(??? : Do[_<: Tape[Double,Double]])" should compile
      "abs(??? : Do[Tape[Double,Double]])" should compile
      "abs(??? : Do[Tape[Double, Double]])" should compile
      "abs(??? : Do[_<: Tape[Double, Double]])" should compile
      "abs(??? : Do[Tape[Double, Double]])" should compile

      "abs(Nd4j.ones(2, 3, 3, 3))" should compile
      "abs(??? : Do[Tape[INDArray,INDArray]])" should compile
      "abs(??? : Do[_<: Tape[INDArray,INDArray]])" should compile
      "abs(??? : Do[Tape[INDArray,INDArray]])" should compile
      "abs(??? : Do[Tape[INDArray, INDArray]])" should compile
      "abs(??? : Do[_<: Tape[INDArray, INDArray]])" should compile
      "abs(??? : Do[Tape[INDArray, INDArray]])" should compile
    }

    promise.future
  }
}
