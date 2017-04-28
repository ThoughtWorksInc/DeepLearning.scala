package com.thoughtworks.deeplearning.differentiable

import com.thoughtworks.deeplearning.{Tape, ToTapeTask}
import com.thoughtworks.deeplearning.differentiable.float.Optimizer
import com.thoughtworks.deeplearning.differentiable.int.Optimizer.LearningRate
import com.thoughtworks.deeplearning.differentiable.int.Weight
import com.thoughtworks.deeplearning.differentiable.int.implicits._
import com.thoughtworks.deeplearning.differentiable.int._
import com.thoughtworks.each.Monadic._
import com.thoughtworks.raii.future.Do
import com.thoughtworks.raii.transformers.ResourceFactoryT
import org.scalatest.{Assertion, AsyncFreeSpec, Inside, Matchers}
import com.thoughtworks.deeplearning.PolyFunctions._
import com.thoughtworks.deeplearning.Tape.Aux
import com.thoughtworks.deeplearning.TapeTask.{predict, train}
import com.thoughtworks.raii.future

import scala.concurrent.Promise
import scalaz.{-\/, \/, \/-}
import scalaz.concurrent.{Future, Task}
import scalaz.std.`try`.toDisjunction
import scalaz.syntax.all._
import scalaz.std.iterable._

class intSpec extends AsyncFreeSpec with Matchers with Inside {
  "int" in {

    implicit def optimizer: Optimizer = new LearningRate {
      def currentLearningRate() = 1
    }

    val weight: Weight = 1.toWeight

    def myNetwork(input: Int): Do[Tape.Aux[Int, Float]] = {
      -10 + 20 - ((input - weight + 4) * 2 / 2)
    }

    def trainMyNetwork(inputData: Int): Task[Int] = {
      train(myNetwork(inputData))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.future.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 6) {
        trainMyNetwork(1).each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      predict(myNetwork(1)).each
    }

    val p = Promise[Assertion]
    result.unsafePerformAsync { either: \/[Throwable, Int] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) => {
            loss should be(0)
            weight.data should be(-5)
          }
        }
      }
    }
    p.future
  }
}
