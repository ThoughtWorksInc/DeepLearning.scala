package com.thoughtworks.deeplearning

import cats.kernel.CommutativeGroup
import com.qifun.statelessFuture.Future
import com.thoughtworks.deeplearning.CumulativeTape.MonoidTape
import com.thoughtworks.deeplearning.Layer.Tape
import shapeless.the

/**
  * A namespace of common operators for Double layers.
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableFloat {

  object Tapes {
    final class Plus(upstream0: Tape.Aux[Float, Float], upstream1: Tape.Aux[Float, Float])
        extends CumulativeTape
        with MonoidTape {

      override type Data = Float
      override type Delta = Float

      override protected def closeUpstreams(): Future[Unit] = Future {
        // TODO: parallelize the two close calls
        upstream0.close().await
        upstream1.close().await
      }

      override val isTrainable: Boolean = upstream0.isTrainable || upstream1.isTrainable

      override val value: Float = upstream0.value + upstream1.value

      override protected def rawBackward(delta: Float): Future[Unit] = Future {
        // TODO: parallelize the two backward calls
        upstream0.backward(delta).await
        upstream1.backward(delta).await
      }

      override implicit protected def monoid: CommutativeGroup[Float] = cats.instances.float.catsKernelStdGroupForFloat
    }

    final class Negative(upstream: Tape.Aux[Float, Float]) extends CumulativeTape with MonoidTape {

      override type Data = Float
      override type Delta = Float

      override protected def closeUpstreams(): Future[Unit] = Future {
        upstream.close().await
      }

      /**
        * Performs the underlying backward pass with all `outputDelta`s that previously received from [[forceBackward]].
        */
      override protected def rawBackward(delta: Float): Future[Unit] = Future {
        upstream.backward(-delta)
      }

      override implicit protected def monoid: CommutativeGroup[Float] = cats.instances.float.catsKernelStdGroupForFloat

      override def isTrainable: Boolean = upstream.isTrainable

      override def value: Float = -upstream.value
    }

    final class Reciprocal(upstream: Tape.Aux[Float, Float]) extends CumulativeTape with MonoidTape {
      override type Data = Float
      override type Delta = Float

      override protected def closeUpstreams(): Future[Unit] = Future {
        upstream.close().await
      }

      /**
        * Performs the underlying backward pass with all `outputDelta`s that previously received from [[forceBackward]].
        */
      override protected def rawBackward(delta: Float): Future[Unit] = Future {
        val a = upstream.value
        upstream.backward(-delta / (a * a))
      }

      override implicit protected def monoid: CommutativeGroup[Float] = cats.instances.float.catsKernelStdGroupForFloat

      override def isTrainable: Boolean = upstream.isTrainable

      override def value: Float = the[Numeric[Float]].one / upstream.value
    }

    final class Substract(upstream0: Tape.Aux[Float, Float], upstream1: Tape.Aux[Float, Float])
        extends CumulativeTape
        with MonoidTape {

      override type Data = Float
      override type Delta = Float

      override protected def closeUpstreams(): Future[Unit] = Future {
        upstream0.close().await
        upstream1.close().await
      }

      /**
        * Performs the underlying backward pass with all `outputDelta`s that previously received from [[forceBackward]].
        */
      override protected def rawBackward(delta: Float): Future[Unit] = Future {
        upstream0.backward(delta)
        upstream1.backward(-delta)
      }

      override implicit protected def monoid: CommutativeGroup[Float] = cats.instances.float.catsKernelStdGroupForFloat

      override def isTrainable: Boolean = upstream0.isTrainable || upstream1.isTrainable

      override def value: Float = upstream0.value - upstream1.value
    }

    final class Times(upstream0: Tape.Aux[Float, Float], upstream1: Tape.Aux[Float, Float])
        extends CumulativeTape
        with MonoidTape {

      override type Data = Float
      override type Delta = Float

      override protected def closeUpstreams(): Future[Unit] = Future {
        upstream0.close().await
        upstream1.close().await
      }

      /**
        * Performs the underlying backward pass with all `outputDelta`s that previously received from [[forceBackward]].
        */
      override protected def rawBackward(delta: Float): Future[Unit] = Future {
        val a = upstream0.value
        val b = upstream1.value
        upstream0.backward(delta * b)
        upstream1.backward(delta * a)
      }

      override implicit protected def monoid: CommutativeGroup[Float] = cats.instances.float.catsKernelStdGroupForFloat

      override def isTrainable: Boolean = upstream0.isTrainable || upstream1.isTrainable

      override def value: Float = upstream0.value * upstream1.value
    }

    final class Log(upstream: Tape.Aux[Float, Float]) extends CumulativeTape with MonoidTape {

      override type Data = Float
      override type Delta = Float

      override protected def closeUpstreams(): Future[Unit] = Future {
        upstream.close().await
      }

      /**
        * Performs the underlying backward pass with all `outputDelta`s that previously received from [[forceBackward]].
        */
      override protected def rawBackward(delta: Float): Future[Unit] = Future {
        upstream.backward(delta / upstream.value)
      }

      override implicit protected def monoid: CommutativeGroup[Float] = cats.instances.float.catsKernelStdGroupForFloat

      override def isTrainable: Boolean = upstream.isTrainable

      override def value: Float = math.log(upstream.value).toFloat
    }

    final class Exp(upstream: Tape.Aux[Float, Float]) extends CumulativeTape with MonoidTape {

      override type Data = Float
      override type Delta = Float

      override protected def closeUpstreams(): Future[Unit] = Future {
        upstream.close().await
      }

      /**
        * Performs the underlying backward pass with all `outputDelta`s that previously received from [[forceBackward]].
        */
      override protected def rawBackward(delta: Float): Future[Unit] = Future {
        upstream.backward(delta * value)
      }

      override implicit protected def monoid: CommutativeGroup[Float] = cats.instances.float.catsKernelStdGroupForFloat

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
