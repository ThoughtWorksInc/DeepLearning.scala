package com.thoughtworks.deeplearning
package differentiable

import java.util.logging.Level

import com.thoughtworks.deeplearning.PolyFunctions._
import com.thoughtworks.deeplearning.Tape.{Aux, Literal}
import com.thoughtworks.deeplearning.TapeTask.{predict, train}
import com.thoughtworks.deeplearning.TapeTaskFactory.BinaryTapeTaskFactory.MultipleException
import com.thoughtworks.deeplearning.differentiable.float.Optimizer._
import com.thoughtworks.deeplearning.differentiable.float._
import com.thoughtworks.deeplearning.differentiable.float.implicits._
import com.thoughtworks.each.Monadic._
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import com.thoughtworks.raii.resourcet
import com.thoughtworks.raii.resourcet.{ResourceT, Releasable}
import com.thoughtworks.tryt.{TryT, TryTExtractor}
import org.scalactic.ErrorMessage
import org.scalatest._
import org.slf4j.bridge.SLF4JBridgeHandler
import com.thoughtworks.raii.asynchronous.Do._
import com.thoughtworks.raii.ownership.Borrowing

import scalaz.concurrent.Future.futureInstance
import scala.concurrent.{ExecutionContext, Promise}
import scala.util.Try
import scalaz.concurrent.{Future, Task}
import scalaz.std.option._
import scalaz.{-\/, EitherT, MonadError, \/, \/-}
import scalaz.syntax.all._
import scalaz.std.`try`.toDisjunction
import scalaz.std.iterable._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object floatSpec {
  case class Boom(errorMessage: ErrorMessage) extends RuntimeException

  private def throwableFloatTapeTask(throwable: Throwable): Do[Borrowing[Tape.Aux[Float, Float]]] = {
    import com.thoughtworks.raii.resourcet.ResourceT._
    import scalaz.concurrent.Future._

    val value1: TryT[ResourceT[Future, ?], Borrowing[Tape.Aux[Float, Float]]] = TryT
      .tryTMonadError[ResourceT[Future, ?]]
      .raiseError[Borrowing[Tape.Aux[Float, Float]]](throwable)

    val value3: ResourceT[Future, Try[Borrowing[Tape.Aux[Float, Float]]]] =
      TryT.unapply[ResourceT[Future, ?], Borrowing[Tape.Aux[Float, Float]]](value1).get

    val value2: Future[Releasable[Future, Try[Borrowing[Tape.Aux[Float, Float]]]]] =
      ResourceT.unapply(value3).get

    Do[Borrowing[Tape.Aux[Float, Float]]](value2)
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

  implicit final class WeightData(weight: Do[Borrowing[Tape.Aux[Float, Float]]]) {
    def data: Float = {
      val task: Task[Borrowing[Aux[Float, Float]]] = Do.run(weight)
      val bTape: Borrowing[Tape.Aux[Float, Float]] = task.unsafePerformSync
      bTape.data.toString.toFloat
    }
  }
}

final class floatSpec extends AsyncFreeSpec with Matchers with Inside {
  import floatSpec._
  implicit val logger: java.util.logging.Logger = java.util.logging.Logger.getLogger("floatSpec")
  logger.setLevel(Level.ALL)
  SLF4JBridgeHandler.install()

  "Plus" in {

    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1.0f
    }

    val weight: Do[Borrowing[Tape.Aux[Float, Float]]] = 1.0f.toWeight

    def myNetwork(input: Do[_ <: Borrowing[Tape.Aux[Float, Float]]]): Do[Borrowing[Tape.Aux[Float, Float]]] = {
      6.7f + input + weight + 5.5f
    }

    def train(inputData: Float): Task[Unit] = {
      import com.thoughtworks.raii.resourcet.ResourceT._
      import scalaz.concurrent.Future._
      import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances
      val c: Do[Unit] = myNetwork(ToTapeTask[Float].apply(inputData)).flatMap { outputTape: Tape.Aux[Float, Float] =>
        Do.delay(outputTape.backward(Do.now(1.0f)))
      }

      val Do(futureAsyncReleasable) = c

      val map: Future[\/[Throwable, Unit]] = ResourceT.run(ResourceT(futureAsyncReleasable)).map {
        toDisjunction
      }
      new Task(map)
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
            true should be(true)
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

    val weight: Do[Borrowing[Tape.Aux[Float, Float]]] = 1.0f.toWeight

    def myNetwork(input: Float): Do[Borrowing[Tape.Aux[Float, Float]]] = {
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

    val weight: Do[Borrowing[Tape.Aux[Float, Float]]] = 1.0f.toWeight

    def myNetwork(input: Float): Do[Borrowing[Tape.Aux[Float, Float]]] = {
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

    val weight: Do[Borrowing[Tape.Aux[Float, Float]]] = 1.0f.toWeight

    def myNetwork(input: Float): Do[Borrowing[Tape.Aux[Float, Float]]] = {
      -10.0f + 20.0f - ((input - weight + 4.0f) * 2.0f / 2.0f)
      //10.0f - (input - weight + 4.0f) //6
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      train(myNetwork(inputData))
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

    val weight: Do[Borrowing[Tape.Aux[Float, Float]]] = 1.0f.toWeight

    def myNetwork(input: Float): Do[Borrowing[Tape.Aux[Float, Float]]] = {
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

    val weight: Do[Borrowing[Tape.Aux[Float, Float]]] = 1.0f.toWeight

    def myNetwork(input: Float): Do[Borrowing[Tape.Aux[Float, Float]]] = {
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

    val weight: Do[Borrowing[Tape.Aux[Float, Float]]] = 1.0f.toWeight

    def myNetwork(input: Float): Do[Borrowing[Tape.Aux[Float, Float]]] = {
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

    val weight: Do[Borrowing[Tape.Aux[Float, Float]]] = 1.0f.toWeight

    def myNetwork(input: Float): Do[Borrowing[Tape.Aux[Float, Float]]] = {
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

    val weight: Do[Borrowing[Tape.Aux[Float, Float]]] = 1.0f.toWeight

    def myNetwork(input: Float): Do[Borrowing[Tape.Aux[Float, Float]]] = {
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

    val weight: Do[Borrowing[Tape.Aux[Float, Float]]] = 1.0f.toWeight

    def myNetwork(input: Float): Do[Borrowing[Tape.Aux[Float, Float]]] = {
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

    val weight: Do[Borrowing[Tape.Aux[Float, Float]]] = 1.0f.toWeight

    val log5 = math.log(5).toFloat

    def myNetwork(input: Float): Do[Borrowing[Tape.Aux[Float, Float]]] = {
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

    val weight: Do[Borrowing[Tape.Aux[Float, Float]]] = 1.0f.toWeight

    val exp3 = math.exp(3).toFloat

    def myNetwork(input: Float): Do[Borrowing[Tape.Aux[Float, Float]]] = {
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

    val weight: Do[Borrowing[Tape.Aux[Float, Float]]] = 1.0f.toWeight

    def myNetwork(input: Float): Do[Borrowing[Tape.Aux[Float, Float]]] = {
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

    val weight: Do[Borrowing[Tape.Aux[Float, Float]]] = 5.0f.toWeight

    def myNetwork(input: Float): Do[Borrowing[Tape.Aux[Float, Float]]] = {
      abs(-weight)
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
