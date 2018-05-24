package com.thoughtworks.deeplearning
import java.io.{PrintStream, PrintWriter}

import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.continuation._
import com.thoughtworks.future._
import scalaz.syntax.all._
import com.thoughtworks.raii.asynchronous._
import simulacrum.typeclass

import scala.language.implicitConversions
import algebra.ring.MultiplicativeMonoid
import scalaz.Semigroup

object DeepLearning {

  implicit object multipleExceptionThrowableSemigroup extends Semigroup[Throwable] {
    override def append(f1: Throwable, f2: => Throwable): Throwable =
      f1 match {
        case me1: AbstractMultipleException =>
          f2 match {
            case me2: AbstractMultipleException => MultipleException(me1.throwableSet ++ me2.throwableSet)
            case e: Throwable                   => MultipleException(me1.throwableSet + e)
          }
        case _: Throwable =>
          f2 match {
            case me2: AbstractMultipleException => MultipleException(me2.throwableSet + f1)
            case `f1`                           => f1
            case e: Throwable                   => MultipleException(Set(f1, e))
          }
      }
  }

  private final case class MultipleException(throwableSet: Set[Throwable])
      extends DeepLearning.AbstractMultipleException

  abstract class AbstractMultipleException extends RuntimeException("Multiple exceptions found") {

    def throwableSet: Set[Throwable]

    override def toString: String = throwableSet.mkString("\n")

    override def printStackTrace(): Unit = {
      for (throwable <- throwableSet) {
        throwable.printStackTrace()
      }
    }

    override def printStackTrace(s: PrintStream): Unit = {
      for (throwable <- throwableSet) {
        throwable.printStackTrace(s)
      }
    }

    override def printStackTrace(s: PrintWriter): Unit = {
      for (throwable <- throwableSet) {
        throwable.printStackTrace(s)
      }
    }

    override def getStackTrace: Array[StackTraceElement] = synchronized {
      super.getStackTrace match {
        case null =>
          setStackTrace(throwableSet.flatMap(_.getStackTrace)(collection.breakOut))
          super.getStackTrace
        case stackTrace =>
          stackTrace
      }
    }

    override def fillInStackTrace(): this.type = {
      this
    }

  }

  /** The node of wengert list created during [[DeepLearning.forward forward]] pass */
  final case class Tape[+Data, -Delta](data: Data, backward: Do[Delta] => UnitContinuation[Unit])

  type Aux[Differentiable, Data0, Delta0] = DeepLearning[Differentiable] {
    type Data = Data0
    type Delta = Delta0
  }

  private[DeepLearning] sealed trait SimulacrumIssue82WorkAround[Differentiable] {

    /** The result value of forward pass */
    type Data

    /** The partial derivative for [[Data]] */
    type Delta

    /** Returns an asynchronous [[com.thoughtworks.raii.asynchronous.Do Do]] of forward pass, which creates a wengert list. */
    def forward(differentiable: Differentiable): Do[Tape[Data, Delta]]

    /** Returns a [[com.thoughtworks.future.Future Future]] that updates [[plugins.Weights.Weight Weight]] internally used by `differentiable`. */
    def train(differentiable: Differentiable)(implicit monoid: MultiplicativeMonoid[Delta]): Future[Data]

    /** Returns a [[com.thoughtworks.future.Future Future]] of the [[DeepLearning.Tape.data data]] of the `differentiable` expression. */
    def predict(differentiable: Differentiable): Future[Data]
  }

}
import DeepLearning._

/** A type class that witnesses `Differentiable` is a differentiable expression.
  *
  * Common differentiable types that supports [[DeepLearning]] are:
  *
  *  - [[scala.Float Float]], [[plugins.FloatWeights.FloatWeight FloatWeight]] or [[plugins.CumulativeFloatLayers.FloatLayer FloatLayer]]
  *  - [[scala.Double Double]], [[plugins.DoubleWeights.DoubleWeight DoubleWeight]] or [[plugins.CumulativeDoubleLayers.DoubleLayer DoubleLayer]]
  *  - [[org.nd4j.linalg.api.ndarray.INDArray INDArray]], [[plugins.INDArrayWeights.INDArrayWeight INDArrayWeight]] or [[plugins.CumulativeINDArrayLayers.INDArrayLayer INDArrayLayer]]
  *
  * @note The Scaladoc of members of trait DeepLearning are defined inside `SimulacrumIssue82WorkAround`,
  *       in case of https://github.com/mpilquist/simulacrum/issues/82
  */
@typeclass(excludeParents = List("SimulacrumIssue82WorkAround"))
trait DeepLearning[Differentiable] extends SimulacrumIssue82WorkAround[Differentiable] {

  type Data
  type Delta

  def forward(differentiable: Differentiable): Do[Tape[Data, Delta]]

  final def train(differentiable: Differentiable)(implicit monoid: MultiplicativeMonoid[Delta]): Future[Data] = {
    val doData = forward(differentiable).flatMap[Data] { tape =>
      Do.garbageCollected(tape.backward(Do.now(monoid.one))).intransitiveMap { _: Unit =>
        tape.data
      }
    }
    doData.run
  }

  final def predict(differentiable: Differentiable): Future[Data] = {
    val doData = forward(differentiable).map(_.data)
    doData.run
  }
}
