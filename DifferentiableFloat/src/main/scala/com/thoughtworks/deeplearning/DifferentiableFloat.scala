package com.thoughtworks.deeplearning

import com.qifun.statelessFuture.Future
import com.thoughtworks.deeplearning.CumulativeTape.MonoidTape
import com.thoughtworks.deeplearning.Layer.Tape

/**
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

      override implicit protected def monoid = cats.instances.float.catsKernelStdGroupForFloat
    }
  }

  import Tapes._

  implicit final class FloatTapeOps(floatTape: Tape.Aux[Float, Float]) {
    def +(right: Tape.Aux[Float, Float]) = {
      new Plus(floatTape.duplicate(), right.duplicate())
    }
  }

  // implicit helpers, ops, ...
}
