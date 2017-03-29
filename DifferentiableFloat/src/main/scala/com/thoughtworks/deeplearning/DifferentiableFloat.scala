package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.CumulativeTape.MonoidTape
import com.thoughtworks.deeplearning.Layer.Tape
import shapeless.the

import scala.util.Try
import scala.util.control.Exception.Catcher
import scala.util.control.TailCalls.TailRec
import cats.Monoid
import com.thoughtworks.future.Continuation.Task
import com.thoughtworks.future.Future

import scalaz.syntax.zip._
import com.thoughtworks.future.scalaz.TaskInstance.scalazTaskInstance

/**
  * A namespace of common operators for Float layers.
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableFloat {

  object Tapes {

    private[Tapes] trait FloatMonoidTape extends MonoidTape { this: Tape =>

      override final type Data = Float

      override final type Delta = Float

      override protected final def monoid: Monoid[Float] = cats.instances.float.catsKernelStdGroupForFloat
    }
    trait Plus
//
//    trait Plus extends FloatMonoidTape with BinaryTape { this: Tape =>
//      override final type Upstream0 = Tape.Aux[Float, Float]
//      override final type Upstream1 = Tape.Aux[Float, Float]
//      override final val value: Float = upstream0.value + upstream1.value
//      override protected final def upstreamDelta(outputDelta: Delta) = {
//        val upstream0Delta = future { outputDelta }
//        val upstream1Delta = future { outputDelta }
//        (upstream0Delta, upstream1Delta)
//      }
//
//    }
  }

  import Tapes._

  private type FloatFuture = Future[Tape.Aux[Float, Float]]

  implicit final class FloatTapeOps(operand0: Task[Tape.Aux[Float, Float]]) {
    def +(operand1: Task[Tape.Aux[Float, Float]]): FloatFuture = Future.completeWith {
      operand0.fzipWith(operand1) { (upstream0: Tape.Aux[Float, Float], upstream1: Tape.Aux[Float, Float]) =>
//        CumulativeTape[Plus](upstream0, upstream1)
        ???
      }

      //CumulativeTape.makeFuture[Plus](operand0, operand1)
    }

  }
  // implicit helpers, ops, ...
}

/*

val a: Future[...] = b + c

val d = a + a


def train(f: Future[Tape...]) = {

  f.onComplete { t: Tape =>
    t.retain()
    try {
      t.backward(t.value)
    } finally {
      t.release()
    }
  }

}
 */
