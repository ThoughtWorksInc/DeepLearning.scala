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
object DifferentiableInt {
  private[deeplearning] trait IntMonoidTape extends CumulativeTape {

    override type Data = Int

    override type Delta = Float

    protected final def monoid: Monoid[Float] = implicitly[Monoid[Delta]]
  }

  val optimizers = DifferentiableFloat.Optimizers

  abstract case class Weight(var value: Int) extends Layer with Tape {

    override type Data = Int
    override type Delta = Float

    override type Input = Tape
    override type Output = Tape.Aux[Data, Delta]

    override final def isTrainable = true

    protected def optimizer: Optimizer

    override final def forward(any: Input) = Future {
      this
    }

    override final def duplicate(): Weight = this

    override def backward(delta: Delta): Future[Unit] = Future {
      synchronized {
        value = optimizer.updateFloat(value, delta).toInt
      }
    }

    override final def close(): Future[Unit] = Future {}
  }

  object Tapes {
    final class Plus(upstream0: Tape.Aux[Int, Float], upstream1: Tape.Aux[Int, Float])
        extends IntMonoidTape
        with MonoidTape {

      override protected def closeUpstreams(): Future[Unit] = Future {
        // TODO: parallelize the two close calls
        upstream0.close().await
        upstream1.close().await
      }

      override val isTrainable: Boolean = upstream0.isTrainable || upstream1.isTrainable

      override val value: Int = upstream0.value + upstream1.value

      override protected def flush(delta: Float): Future[Unit] = Future {
        // TODO: parallelize the two backward calls
        upstream0.backward(delta).await
        upstream1.backward(delta).await
      }
    }

    final class Negative(upstream: Tape.Aux[Int, Float]) extends IntMonoidTape with MonoidTape {

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

      override def value: Int = -upstream.value
    }

    final class Reciprocal(upstream: Tape.Aux[Int, Float]) extends IntMonoidTape with MonoidTape {

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

      override def value: Int = the[Numeric[Int]].one / upstream.value
    }

    final class Substract(upstream0: Tape.Aux[Int, Float], upstream1: Tape.Aux[Int, Float])
        extends IntMonoidTape
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

      override def value: Int = upstream0.value - upstream1.value
    }

    final class Times(upstream0: Tape.Aux[Int, Float], upstream1: Tape.Aux[Int, Float])
        extends IntMonoidTape
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

      override def value: Int = upstream0.value * upstream1.value
    }

    final class Log(upstream: Tape.Aux[Int, Float]) extends IntMonoidTape with MonoidTape {

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

      override def value: Int = math.log(upstream.value).toInt
    }

    final class Exp(upstream: Tape.Aux[Int, Float]) extends IntMonoidTape with MonoidTape {

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

      override def value: Int = math.exp(upstream.value).toInt
    }
  }

  import Tapes._

  implicit final class IntTapeOps(IntTape: Tape.Aux[Int, Float]) {
    def +(right: Tape.Aux[Int, Float]): Plus = {
      new Plus(IntTape.duplicate(), right.duplicate())
    }

    def - : Negative = {
      new Negative(IntTape)
    }

    def *(right: Tape.Aux[Int, Float]): Times = {
      new Times(IntTape, right)
    }

    def /(right: Tape.Aux[Int, Float]): Times = {
      new Times(IntTape, new Reciprocal(right))
    }

    def log: Log = {
      new Log(IntTape)
    }

    def exp: Exp = {
      new Exp(IntTape)
    }

  }
}
