package com.thoughtworks.deeplearning
import com.thoughtworks.deeplearning.DeepLearning.Tape

import scalaz.concurrent.{Future, Task}
import scalaz.syntax.all._
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import simulacrum.typeclass

import scala.language.implicitConversions
import spire.algebra.MultiplicativeMonoid

object DeepLearning {

  /** The node of wengert list created during [[DeepLearning.forward forward]] pass */
  final case class Tape[+Data, -Delta](data: Data, backward: Do[Delta] => Future[Unit])

  type Aux[Differentiable, Data0, Delta0] = DeepLearning[Differentiable] {
    type Data = Data0
    type Delta = Delta0
  }

  // The Scaladoc of members of trait DeepLearning must defined in `SimulacrumIssue82WorkAround`,
  // in case of https://github.com/mpilquist/simulacrum/issues/82
  private[DeepLearning] sealed trait SimulacrumIssue82WorkAround[Differentiable] {

    /** The result value of forward pass */
    type Data

    /** The partial derivative for [[Data]] */
    type Delta

    /** Returns an asynchronous operation of forward pass, which creates a wengert list. */
    def forward(differentiable: Differentiable): Do[Tape[Data, Delta]]

    /** Returns a [[scalaz.concurrent.Task Task]] that updates [[plugins.Weights.Weight Weight]] internally used by `differentiable`. */
    def train(differentiable: Differentiable)(implicit monoid: MultiplicativeMonoid[Delta]): Task[Data]

    /** Returns a [[scalaz.concurrent.Task Task]] of the value of the `differentiable` expression. */
    def predict(differentiable: Differentiable): Task[Data]
  }

}
import DeepLearning._

/** A type class that witnesses `Differentiable` is a differentiable expression.
  *
  * Common differentiable types that supports [[DeepLearning]] are:
  *
  *  - [[scala.Float Float]], [[plugins.FloatWeights.FloatWeight FloatWeight]] or [[plugins.FloatLayers.FloatLayer FloatLayer]]
  *  - [[scala.Double Double]], [[plugins.DoubleWeights.DoubleWeight DoubleWeight]] or [[plugins.DoubleLayers.DoubleLayer DoubleLayer]]
  *  - [[org.nd4j.linalg.api.ndarray.INDArray INDArray]], [[plugins.INDArrayWeights.INDArrayWeight INDArrayWeight]] or [[plugins.INDArrayLayers.INDArrayLayer INDArrayLayer]]
  */
@typeclass(excludeParents = List("SimulacrumIssue82WorkAround"))
trait DeepLearning[Differentiable] extends SimulacrumIssue82WorkAround[Differentiable] {

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
