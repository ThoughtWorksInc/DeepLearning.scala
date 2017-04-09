package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Compute._

import scalaz.Monoid
import scalaz.concurrent.{Future, Task}

/**
  * A namespace of common operators for Float layers.
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableFloat {

  private implicit object FloatMonoid extends Monoid[Float] {
    override def zero: Float = 0.0f

    override def append(f1: Float, f2: => Float): Float = f1 + f2
  }

  implicit final class FloatTapeOps(operand0: Compute[Tape.Aux[Float, Float]]) {
    def +(operand1: Compute[Tape.Aux[Float, Float]]): Compute[Tape.Aux[Float, Float]] = {
      Compute.binary(operand0, operand1) { (data0, data1) =>
        Task.delay {
          val outputData = data0 + data1
          def computeDeltas(delta: Float) = {
            val delta0Future = Future.now(delta)
            val delta1Future = Future.now(delta)
            (delta0Future, delta1Future)
          }
          (outputData, computeDeltas)
        }
      }
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
