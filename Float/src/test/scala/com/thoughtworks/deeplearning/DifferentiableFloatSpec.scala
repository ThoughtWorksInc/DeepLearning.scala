package com.thoughtworks.deeplearning

import org.scalatest._
import com.thoughtworks.deeplearning.Float._
import com.thoughtworks.deeplearning.Tape.{Aux, Literal}
import PolyFunctions._
import com.thoughtworks.deeplearning.Float.Optimizers.{LearningRate, Optimizer}
import com.thoughtworks.raii.{RAIIFuture, RAIITask, ResourceFactoryT}
import com.thoughtworks.deeplearning.Float.trainableFloat
import com.thoughtworks.deeplearning.Float.toFloatTapeTask
import com.thoughtworks.deeplearning.TapeTask.train
import com.thoughtworks.deeplearning.TapeTask.predict

import scala.concurrent.Promise
import scalaz.concurrent.{Future, Task}
import com.thoughtworks.each.Monadic._
import com.thoughtworks.raii.ResourceFactoryT.ResourceT
import shapeless.Lazy

import scala.annotation.tailrec
import scalaz.{-\/, Foldable, \/, \/-}
import scalaz.std.option._
import scalaz.std.iterable._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DifferentiableFloatSpec extends AsyncFreeSpec with Matchers with Inside {

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

    val t5: Task[Unit] = throwableMonadic[Task] {
      train(1.0f).each
      train(1.0f).each
      train(1.0f).each
      train(1.0f).each
      train(1.0f).each
    }

    val p = Promise[Assertion]
    t5.unsafePerformAsync { either: \/[Throwable, Unit] =>
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

    val t5: Task[Unit] = throwableMonadic[Task] {
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
    }

    val p = Promise[Assertion]
    t5.unsafePerformAsync { either: \/[Throwable, Unit] =>
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

    val t5: Task[Unit] = throwableMonadic[Task] {
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
    }

    val result = throwableMonadic[Task] {
      predict(myNetwork(1.0f)).each
    }

    t5.unsafePerformAsync { _: \/[Throwable, Unit] =>
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
      10.0f - ((input - weight + 4.0f) * 2.0f / 2.0f)
      //10.0f - (input - weight + 4.0f) //6
    }

    def trainMyNetwork(inputData: Float): Task[Float] = {
      train(myNetwork(inputData))
    }

    @monadic[Task]
    val t5: Task[Unit] = {
      for (_ <- 1 to 6) {
        trainMyNetwork(1.0f).each
      }
    }

    val result = throwableMonadic[Task] {
      predict(myNetwork(1.0f)).each
    }

    t5.unsafePerformAsync { _: \/[Throwable, Unit] =>
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

}
