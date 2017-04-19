package com.thoughtworks.deeplearning

import java.util.logging.Level

import org.scalatest._
import com.thoughtworks.deeplearning.Float.{FloatOps, TrainableFloat, toFloatTapeTask, _}
import com.thoughtworks.deeplearning.Tape.{Aux, Literal}
import PolyFunctions._
import com.thoughtworks.deeplearning.Float.Optimizers.{L1Regularization, LearningRate, Optimizer}
import com.thoughtworks.raii.{RAIIFuture, RAIITask, ResourceFactoryT}
import com.thoughtworks.deeplearning.TapeTask.train
import com.thoughtworks.deeplearning.TapeTask.predict
import com.thoughtworks.deeplearning.TapeTaskFactory.BinaryTapeTaskFactory.MultipleException

import scala.concurrent.{ExecutionContext, Promise}
import scalaz.concurrent.{Future, Task}
import com.thoughtworks.each.Monadic._
import com.thoughtworks.raii.ResourceFactoryT.ResourceT
import org.scalactic.ErrorMessage
import org.slf4j.bridge.SLF4JBridgeHandler
import shapeless.Lazy

import scala.annotation.tailrec
import scalaz.{-\/, EitherT, Foldable, \/, \/-}
import scalaz.std.option._
import scalaz.std.iterable._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class FloatSpec extends AsyncFreeSpec with Matchers with Inside {

  case class Boom(errorMessage: ErrorMessage) extends RuntimeException

  private def throwableFloatTapeTask(throwable: Throwable): RAIITask[Tape.Aux[Float, Float]] = {
    EitherT
      .eitherTMonadError[ResourceFactoryT[Future, ?], Throwable]
      .raiseError[Tape.Aux[Float, Float]](throwable)
  }

  private def jump()(implicit executionContext: ExecutionContext): Task[Unit] = {
    Task.async { handler: ((Throwable \/ Unit) => Unit) =>
      executionContext.execute {
        new Runnable {
          override def run(): Unit = handler(\/-(()))
        }
      }
    }
  }

  implicit val logger: java.util.logging.Logger = java.util.logging.Logger.getLogger("FloatSpec")
  logger.setLevel(Level.ALL)
  SLF4JBridgeHandler.install()

  "Plus" in {

    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1.0f
    }

    val weight: Weight = 1.0f.toWeight

    def myNetwork(input: RAIITask[Tape.Aux[Float, Float]]): RAIITask[Tape.Aux[Float, Float]] = {
      6.7f + input + weight + 5.5f
    }

    def train(inputData: Float): Task[Unit] = {
      val c: RAIITask[Unit] = myNetwork(RAIITask.now(Literal(inputData): Tape.Aux[Float, Float])).flatMap {
        outputTape =>
          RAIITask.unmanaged(outputTape.backward(RAIITask.now(1.0f)))
      }
      new Task(c.run.run)
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
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(_) => {
            weight.data should be(-4)
          }
        }
      }
    }
    p.future
  }

  "Plus with Train" in {

    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1.0f
    }

    val weight: Weight = 1.0f.toWeight

    def myNetwork(input: Float): RAIITask[Tape.Aux[Float, Float]] = {
      6.7f + input + weight + 5.5f
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      train(myNetwork(inputData))
    }

    val task: Task[Unit] = throwableMonadic[Task] {
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
    }

    val p = Promise[Assertion]
    task.unsafePerformAsync { either: \/[Throwable, Unit] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(_) => {
            weight.data should be(-4)
          }
        }
      }
    }
    p.future
  }

  "Plus with Predict" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1.0f
    }

    val weight: Weight = 1.0f.toWeight

    def myNetwork(input: Float): RAIITask[Tape.Aux[Float, Float]] = {
      1.0f + input + weight + 4.0f
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      train(myNetwork(inputData))
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
      predict(myNetwork(1.0f)).each
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
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1.0f
    }

    val weight: Weight = 1.0f.toWeight

    def myNetwork(input: Float): RAIITask[Tape.Aux[Float, Float]] = {
      -10.0f + 20.0f - ((input - weight + 4.0f) * 2.0f / 2.0f)
      //10.0f - (input - weight + 4.0f) //6
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      train(myNetwork(inputData))
    }

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 6) {
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(1.0f)).each
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

  "Predict -- one exception" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1.0f
    }

    val weight: Weight = 1.0f.toWeight

    def myNetwork(input: Float): RAIITask[Tape.Aux[Float, Float]] = {
      10.0f - ((input - weight + throwableFloatTapeTask(Boom("4.0f"))) * 2.0f / 2.0f)
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      train(myNetwork(inputData))
    }

    val p = Promise[Assertion]

    recoverToSucceededIf[MultipleException] {
      p.future
    }

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 6) {
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(1.0f)).each
    }

    result.unsafePerformAsync { either: \/[Throwable, Float] =>
      p.success {
        inside(either) {
          case -\/(e) =>
            e should be(a[Boom])
        }
      }
    }

    p.future
  }

  "Predict -- two exception" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1.0f
    }

    val weight: Weight = 1.0f.toWeight

    def myNetwork(input: Float): RAIITask[Tape.Aux[Float, Float]] = {
      10.0f - ((input - throwableFloatTapeTask(Boom("weight"))
        + throwableFloatTapeTask(Boom("4.0f"))) * 2.0f / 2.0f)
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      train(myNetwork(inputData))
    }

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 6) {
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(1.0f)).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Float] =>
      p.success {
        inside(either) {
          case -\/(e) => {
            e should be(a[MultipleException])
            inside(e) {
              case MultipleException(multipleException) => multipleException.size should be(2)
            }
          }
        }
      }
    }

    p.future
  }

  "Predict -- three exception" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1.0f
    }

    val weight: Weight = 1.0f.toWeight

    def myNetwork(input: Float): RAIITask[Tape.Aux[Float, Float]] = {
      10.0f - ((input - throwableFloatTapeTask(Boom("weight"))
        + throwableFloatTapeTask(Boom("4.0f"))) * 2.0f / throwableFloatTapeTask(Boom("2.0f")))
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      train(myNetwork(inputData))
    }

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 6) {
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(1.0f)).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Float] =>
      p.success {
        inside(either) {
          case -\/(e) => {
            e should be(a[MultipleException])
            inside(e) {
              case MultipleException(multipleException) => multipleException.size should be(3)
            }
          }
        }
      }
    }

    p.future
  }

  "will not stackOverFlow" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1.0f
    }

    val weight: Weight = 1.0f.toWeight

    def myNetwork(input: Float): RAIITask[Tape.Aux[Float, Float]] = {
      -10.0f + 20.0f - ((input - weight + 4.0f) * 2.0f / 2.0f)
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      train(myNetwork(inputData))
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
      predict(myNetwork(1.0f)).each
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
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1.0f
    }

    val weight: Weight = 1.0f.toWeight

    def myNetwork(input: Float): RAIITask[Tape.Aux[Float, Float]] = {
      5.0f - min(5.0f, weight)
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      train(myNetwork(inputData))
    }

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 4) {
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(1.0f)).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Float] =>
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
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1.0f
    }

    val weight: Weight = 1.0f.toWeight

    def myNetwork(input: Float): RAIITask[Tape.Aux[Float, Float]] = {
      10.0f - max(0.0f, weight)
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      train(myNetwork(inputData))
    }

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 9) {
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(1.0f)).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Float] =>
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
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 0.5f
    }

    val weight: Weight = 1.0f.toWeight

    val log5 = math.log(5).toFloat

    def myNetwork(input: Float): RAIITask[Tape.Aux[Float, Float]] = {
      log5 - log(weight)
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      train(myNetwork(inputData))
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
      predict(myNetwork(1.0f)).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Float] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            math.abs(weight.data - 5) should be < 0.1f
            loss should be < 0.1f
        }
      }
    }

    p.future
  }

  "exp" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 0.1f
    }

    val weight: Weight = 1.0f.toWeight

    val exp3 = math.exp(3).toFloat

    def myNetwork(input: Float): RAIITask[Tape.Aux[Float, Float]] = {
      exp3 - exp(weight)
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      train(myNetwork(inputData))
    }

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 4) {
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(1.0f)).each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Float] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            math.abs(weight.data - 3) should be < 0.1f
            loss should be < 0.5f
        }
      }
    }

    p.future
  }

  "abs" in {
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1.0f
    }

    val weight: Weight = 1.0f.toWeight

    def myNetwork(input: Float): RAIITask[Tape.Aux[Float, Float]] = {
      5.0f - abs(weight)
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      train(myNetwork(inputData))
    }

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 4) {
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(1.0f)).each
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
    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1.0f
    }

    val weight: Weight = 5.0f.toWeight

    def myNetwork(input: Float): RAIITask[Tape.Aux[Float, Float]] = {
      import com.thoughtworks.deeplearning.Float.toFloatOps
      abs(-RAIITask.now(weight))
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      train(myNetwork(inputData))
    }

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 5) {
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(1.0f)).each
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
