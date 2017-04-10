package com.thoughtworks.deeplearning

import com.thoughtworks.raii.RAIITask

import scalaz.Monoid
import scalaz.concurrent.{Future, Task}

/**
  * A namespace of common operators for Float layers.
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object Float {

  private implicit object FloatMonoid extends Monoid[Float] {
    override def zero: Float = 0.0f

    override def append(f1: Float, f2: => Float): Float = f1 + f2
  }

  implicit final class FloatComputeOps(operand0: RAIITask[Tape.Aux[Float, Float]]) {
    def +(operand1: RAIITask[Tape.Aux[Float, Float]]): RAIITask[Tape.Aux[Float, Float]] = {
      TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
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
