package com.thoughtworks.deeplearning

import cats.Monoid
import cats.implicits._
import com.qifun.statelessFuture.Future
import com.thoughtworks.deeplearning.CumulativeTape.MonoidTape
import com.thoughtworks.deeplearning.DifferentiableFloat.Optimizers.Optimizer
import com.thoughtworks.deeplearning.Layer.Tape
import shapeless.the

/**
  * A namespace of common operators for Float layers.
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableFloat {

  private[deeplearning] trait FloatMonoidTape extends CumulativeTape {

    override type Data = Float

    override type Delta = Float

    protected final def monoid: Monoid[Float] = implicitly[Monoid[Delta]]
  }

  /**
    * Optimizers of Float.
    *
    * @example{{{
    * implicit val optimizerFactory = new DifferentiableFloat.OptimizerFactory {
    *   override def FloatOptimizer(weight: Weight): Optimizer = {
    *     new LearningRate with L2Regularization {
    *
    *       var learningRate = 0.00003
    *
    *       override protected def l2Regularization: Float = 0.003
    *
    *       override protected def currentLearningRate(): Float = {
    *       learningRate * 0.75
    *       learningRate
    *      }
    *    }
    *  }
    * }
    * }}}
    */
  object Optimizers {

    trait Optimizer {
      def currentDelta(oldValue: Float, delta: Float): Float = delta

      final def updateFloat(oldValue: Float, delta: Float): Float = {
        oldValue - currentDelta(oldValue, delta)
      }
    }

    trait LearningRate extends Optimizer {

      protected def currentLearningRate(): Float

      override def currentDelta(oldValue: Float, delta: Float): Float = delta * currentLearningRate()
    }

    trait L1Regularization extends Optimizer {
      protected def l1Regularization: Float

      override def currentDelta(oldValue: Float, delta: Float): Float = {
        super.currentDelta(oldValue, delta + math.signum(oldValue) * l1Regularization)
      }
    }

    trait L2Regularization extends Optimizer {
      protected def l2Regularization: Float

      override def currentDelta(oldValue: Float, delta: Float): Float = {
        super.currentDelta(oldValue, delta + oldValue * l2Regularization)
      }
    }

  }

  object OptimizerFactory {
    implicit def shared(implicit optimizer: Optimizer): OptimizerFactory = new OptimizerFactory {
      override def floatOptimizer(weight: Weight): Optimizer = optimizer
    }
  }

  trait OptimizerFactory {
    def floatOptimizer(weight: Weight): Optimizer
  }

  object Weight {

    def apply(value: Float)(implicit optimizerFactory: OptimizerFactory) = new Weight(value) {
      override protected val optimizer: Optimizer = optimizerFactory.floatOptimizer(this)
    }

  }

  abstract case class Weight(var value: Float) extends Layer with Tape {

    override type Data = Float
    override type Delta = Float

    override type Input = Tape
    override type Output = Tape.Aux[Data, Delta]

    override final def isTrainable = true

    protected def optimizer: Optimizer

    override final def forward(any: Input) = Future {
      this
    }

    override final def duplicate(): Weight = this

    override def forceBackward(delta: Delta): Future[Unit] = Future {
      synchronized {
        value = optimizer.updateFloat(value, delta)
      }
    }

    override final def close(): Future[Unit] = Future {}
  }

  object Tapes {
    final class Plus(upstream0: Tape.Aux[Float, Float], upstream1: Tape.Aux[Float, Float])
        extends FloatMonoidTape
        with MonoidTape {

      override protected def closeUpstreams(): Future[Unit] = Future {
        // TODO: parallelize the two close calls
        upstream0.close().await
        upstream1.close().await
      }

      override val isTrainable: Boolean = upstream0.isTrainable || upstream1.isTrainable

      override val value: Float = upstream0.value + upstream1.value

      override protected def flush(delta: Float): Future[Unit] = Future {
        // TODO: parallelize the two backward calls
        upstream0.backward(delta).await
        upstream1.backward(delta).await
      }
    }

    final class Negative(upstream: Tape.Aux[Float, Float]) extends FloatMonoidTape with MonoidTape {

      override protected def closeUpstreams(): Future[Unit] = Future {
        upstream.close().await
      }

      /**
        * Performs the underlying backward pass with all `outputDelta`s that previously received from [[forceBackward]].
        */
      override protected def flush(delta: Float): Future[Unit] = Future {
        upstream.backward(-delta)
      }

      override def isTrainable: Boolean = upstream.isTrainable

      override def value: Float = -upstream.value
    }

    final class Reciprocal(upstream: Tape.Aux[Float, Float]) extends FloatMonoidTape with MonoidTape {

      override protected def closeUpstreams(): Future[Unit] = Future {
        upstream.close().await
      }

      /**
        * Performs the underlying backward pass with all `outputDelta`s that previously received from [[forceBackward]].
        */
      override protected def flush(delta: Float): Future[Unit] = Future {
        val a = upstream.value
        upstream.backward(-delta / (a * a))
      }

      override def isTrainable: Boolean = upstream.isTrainable

      override def value: Float = the[Numeric[Float]].one / upstream.value
    }

    final class Substract(upstream0: Tape.Aux[Float, Float], upstream1: Tape.Aux[Float, Float])
        extends FloatMonoidTape
        with MonoidTape {

      override protected def closeUpstreams(): Future[Unit] = Future {
        upstream0.close().await
        upstream1.close().await
      }

      /**
        * Performs the underlying backward pass with all `outputDelta`s that previously received from [[forceBackward]].
        */
      override protected def flush(delta: Float): Future[Unit] = Future {
        upstream0.backward(delta)
        upstream1.backward(-delta)
      }

      override def isTrainable: Boolean = upstream0.isTrainable || upstream1.isTrainable

      override def value: Float = upstream0.value - upstream1.value
    }

    final class Times(upstream0: Tape.Aux[Float, Float], upstream1: Tape.Aux[Float, Float])
        extends FloatMonoidTape
        with MonoidTape {

      override protected def closeUpstreams(): Future[Unit] = Future {
        upstream0.close().await
        upstream1.close().await
      }

      /**
        * Performs the underlying backward pass with all `outputDelta`s that previously received from [[forceBackward]].
        */
      override protected def flush(delta: Float): Future[Unit] = Future {
        val a = upstream0.value
        val b = upstream1.value
        upstream0.backward(delta * b)
        upstream1.backward(delta * a)
      }

      override def isTrainable: Boolean = upstream0.isTrainable || upstream1.isTrainable

      override def value: Float = upstream0.value * upstream1.value
    }

    final class Log(upstream: Tape.Aux[Float, Float]) extends FloatMonoidTape with MonoidTape {

      override protected def closeUpstreams(): Future[Unit] = Future {
        upstream.close().await
      }

      /**
        * Performs the underlying backward pass with all `outputDelta`s that previously received from [[forceBackward]].
        */
      override protected def flush(delta: Float): Future[Unit] = Future {
        upstream.backward(delta / upstream.value)
      }

      override def isTrainable: Boolean = upstream.isTrainable

      override def value: Float = math.log(upstream.value).toFloat
    }

    final class Exp(upstream: Tape.Aux[Float, Float]) extends FloatMonoidTape with MonoidTape {

      override protected def closeUpstreams(): Future[Unit] = Future {
        upstream.close().await
      }

      /**
        * Performs the underlying backward pass with all `outputDelta`s that previously received from [[forceBackward]].
        */
      override protected def flush(delta: Float): Future[Unit] = Future {
        upstream.backward(delta * value)
      }

      override def isTrainable: Boolean = upstream.isTrainable

      override def value: Float = math.exp(upstream.value).toFloat
    }
  }

  import Tapes._

  implicit final class FloatTapeOps(floatTape: Tape.Aux[Float, Float]) {
    def +(right: Tape.Aux[Float, Float]): Plus = {
      new Plus(floatTape.duplicate(), right.duplicate())
    }

    def - : Negative = {
      new Negative(floatTape)
    }

    def *(right: Tape.Aux[Float, Float]): Times = {
      new Times(floatTape, right)
    }

    def /(right: Tape.Aux[Float, Float]): Times = {
      new Times(floatTape, new Reciprocal(right))
    }

    def log: Log = {
      new Log(floatTape)
    }

    def exp: Exp = {
      new Exp(floatTape)
    }

  }

  // implicit helpers, ops, ...
}
