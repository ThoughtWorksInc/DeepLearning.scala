package com.thoughtworks.deeplearning
package plugins

import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.feature.Factory
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.each.Monadic._
import com.thoughtworks.raii.covariant.ResourceT
import org.scalactic.ErrorMessage
import org.scalatest._
 
import scala.concurrent.{ExecutionContext, Promise}
import scala.util.Try
import scalaz.{-\/, \/, \/-}
import scalaz.concurrent.{Future, Task}
import scalaz.std.iterable._

object FloatLayersSpec {

  trait FixedLearningRate extends LearningRate {
    def fixedLearningRate: scala.Float
    trait FloatOptimizerApi extends super.FloatOptimizerApi { this: FloatOptimizer =>
      final def learningRate: scala.Float = fixedLearningRate
    }
    override type FloatOptimizer <: FloatOptimizerApi with Optimizer
  }

  trait LearningRate extends FloatWeights {
    trait FloatOptimizerApi extends super.FloatOptimizerApi { this: FloatOptimizer =>
      def learningRate: scala.Float
      override def delta: scala.Float = super.delta * learningRate
    }
    override type FloatOptimizer <: FloatOptimizerApi with Optimizer
  }

  trait L1Regularization extends FloatWeights {
    def l1Regularization: scala.Float
    trait FloatOptimizerApi extends super.FloatOptimizerApi { this: FloatOptimizer =>
      override def delta: scala.Float = super.delta + scala.math.signum(weight.data) * l1Regularization
    }
    override type FloatOptimizer <: FloatOptimizerApi with Optimizer
  }
  trait L2Regularization extends FloatWeights {
    def l2Regularization: scala.Float
    trait FloatOptimizerApi extends super.FloatOptimizerApi { this: FloatOptimizer =>
      override def delta: scala.Float = super.delta + weight.data * l2Regularization
    }
    override type FloatOptimizer <: FloatOptimizerApi with Optimizer
  }

  case class Boom(errorMessage: ErrorMessage) extends RuntimeException

  private def jump()(implicit executionContext: ExecutionContext): Task[Unit] = {
    Task.async { handler: ((Throwable \/ Unit) => Unit) =>
      executionContext.execute {
        new Runnable {
          override def run(): Unit = handler(\/-(()))
        }
      }
    }
  }
}

/**
  * @author 杨博 (Yang Bo)
  */
final class FloatLayersSpec extends AsyncFreeSpec with Matchers with Inside {
  // TODO: Add tests for exception handling

  import FloatLayersSpec._

  "Plus" in {

    val hyperparameters =
      Factory[FloatTraining with Operators with FloatLiterals with FloatLayers with ImplicitsSingleton with FixedLearningRate]
        .newInstance(fixedLearningRate = 1.0f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      6.7f + input + weight + 5.5f
    }

    def train(inputData: Float): Task[Float] = {
      myNetwork(inputData).train
    }

    val task: Task[Unit] = throwableMonadic[Task] {
      train(1.0f).each
      train(1.0f).each
      train(1.0f).each
      train(1.0f).each
      train(1.0f).each
    }

    val p = Promise[Assertion]

    task.unsafePerformAsync { either: \/[Throwable, Unit] =>
      inside(either) {
        case -\/(e) =>
          p.failure(e)
        case \/-(_) =>
          p.success {
            weight.data should be(-4)
          }
      }
    }

    p.future
  }

  "Plus with Predict" in {
    val hyperparameters = Factory[FloatTraining with Operators with FloatLiterals with FloatLayers with ImplicitsSingleton with FixedLearningRate]
      .newInstance(fixedLearningRate = 1.0f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      1.0f + input + weight + 4.0f
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      myNetwork(inputData).train
    }

    val task: Task[Unit] = throwableMonadic[Task] {
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
    }

    val result = throwableMonadic[Task] {
      task.each
      myNetwork(1.0f).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Float] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss should be(0.0f)
        }
      }
    }

    p.future
  }

  "Predict -- use for" in {
    val hyperparameters = Factory[FloatTraining with Operators with FloatLiterals with FloatLayers with ImplicitsSingleton with FixedLearningRate]
      .newInstance(fixedLearningRate = 1.0f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      -10.0f + 20.0f - ((input - weight + 4.0f) * 2.0f / 2.0f)
      //10.0f - (input - weight + 4.0f) //6
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      myNetwork(inputData).train
    }
    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 6) {
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      myNetwork(1.0f).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Float] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss should be(0.0f)
            weight.data should be(-5)
        }
      }
    }

    p.future
  }

  "will not stackOverFlow" in {
    val hyperparameters =
      Factory[FloatTraining with Operators with FloatLiterals with FloatLayers with ImplicitsSingleton with FixedLearningRate]
        .newInstance(fixedLearningRate = 1.0f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      -10.0f + 20.0f - ((input - weight + 4.0f) * 2.0f / 2.0f)
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      myNetwork(inputData).train
    }

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 1000) {
        Task.apply(()).each
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      myNetwork(1.0f).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Float] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) => true should be(true)
        }
      }
    }

    p.future
  }

  "min" in {
    val hyperparameters =
      Factory[FloatTraining with Operators with FloatLiterals with FloatLayers with ImplicitsSingleton with FixedLearningRate]
        .newInstance(fixedLearningRate = 1.0f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      5.0f - hyperparameters.min(5.0f, weight)
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      myNetwork(inputData).train
    }

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 4) {
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      myNetwork(1.0f).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: Throwable \/ Float =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss should be(0.0f)
            weight.data should be(5)
        }
      }
    }

    p.future
  }

  "max" in {
    val hyperparameters =
      Factory[FloatTraining with Operators with FloatLiterals with FloatLayers with ImplicitsSingleton with FixedLearningRate]
        .newInstance(fixedLearningRate = 1.0f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      10.0f - hyperparameters.max(0.0f, weight)
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      myNetwork(inputData).train
    }

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 9) {
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      myNetwork(1.0f).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: Throwable \/ Float =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss should be(0.0f)
            weight.data should be(10)
        }
      }
    }

    p.future
  }

  "log" in {
    val hyperparameters = Factory[FloatTraining with Operators with FloatLiterals with FloatLayers with ImplicitsSingleton with FixedLearningRate]
      .newInstance(fixedLearningRate = 0.5f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    val log5 = scala.math.log(5).toFloat

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      log5 - hyperparameters.log(weight)
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      myNetwork(inputData).train
    }

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 23) {
        jump().each
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      myNetwork(1.0f).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Float] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            scala.math.abs(weight.data - 5) should be < 0.1f
            loss should be < 0.1f
        }
      }
    }

    p.future
  }

  "exp" in {
    val hyperparameters = Factory[FloatTraining with Operators with FloatLiterals with FloatLayers with ImplicitsSingleton with FixedLearningRate]
      .newInstance(fixedLearningRate = 0.1f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    val exp3 = scala.math.exp(3).toFloat

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      exp3 - hyperparameters.exp(weight)
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      myNetwork(inputData).train
    }

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 4) {
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      myNetwork(1.0f).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Float] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            scala.math.abs(weight.data - 3) should be < 0.1f
            loss should be < 0.5f
        }
      }
    }

    p.future
  }

  "abs" in {
    val hyperparameters =
      Factory[FloatTraining with Operators with FloatLiterals with FloatLayers with ImplicitsSingleton with FixedLearningRate]
        .newInstance(fixedLearningRate = 1.0f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      5.0f - hyperparameters.abs(weight)
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      myNetwork(inputData).train
    }

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 4) {
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      myNetwork(1.0f).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Float] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            weight.data should be(5.0f)
            loss should be(0)
        }
      }
    }
    p.future
  }

  "unary_-" in {
    val hyperparameters =
      Factory[FloatTraining with Operators with FloatLiterals with FloatLayers with ImplicitsSingleton with FixedLearningRate]
        .newInstance(fixedLearningRate = 1.0f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(5.0f)

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      hyperparameters.abs(-weight)
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      myNetwork(inputData).train
    }

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 5) {
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      myNetwork(1.0f).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Float] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            weight.data should be(0.0f)
            loss should be(0)
        }
      }
    }
    p.future
  }
}
