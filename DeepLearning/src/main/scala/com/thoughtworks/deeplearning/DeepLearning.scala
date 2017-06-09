package com.thoughtworks.deeplearning
import scalaz.concurrent.{Future, Task}
import scalaz.syntax.all._
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import simulacrum.typeclass
import scala.language.implicitConversions
import spire.algebra.MultiplicativeMonoid

object DeepLearning {
  final case class Tape[+Data, -Delta](data: Data, backward: Do[Delta] => Future[Unit])

  type Aux[Differentiable, Data0, Delta0] = DeepLearning[Differentiable] {
    type Data = Data0
    type Delta = Delta0
  }
}

/** A type class that witnesses `Differentiable` is a differentiable expression. */
@typeclass
trait DeepLearning[Differentiable] {

  import DeepLearning._

  type Data
  type Delta

  def forward(differentiable: Differentiable): Do[Tape[Data, Delta]]

  final def train(differentiable: Differentiable)(implicit monoid: MultiplicativeMonoid[Delta]): Task[Data] = {
    Do.run(forward(differentiable).flatMap[Data] { tape =>
      Do.delay(tape.backward(Do.now(monoid.one))).map { _ =>
        tape.data
      }
    })
  }

  final def predict(differentiable: Differentiable): Task[Data] = {
    Do.run(forward(differentiable).map(_.data))
  }
}
