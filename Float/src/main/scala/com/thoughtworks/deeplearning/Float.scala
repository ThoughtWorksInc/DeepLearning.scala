package com.thoughtworks.deeplearning

import com.thoughtworks.raii.{RAIIFuture, RAIITask, ResourceFactoryT}
import com.thoughtworks.deeplearning.Float.Optimizers.Optimizer
import com.thoughtworks.deeplearning.Poly.{MathMethods, ToRAIITask}
import com.thoughtworks.deeplearning.Poly.MathMethods._
import com.thoughtworks.deeplearning.Tape.Aux
import com.thoughtworks.raii.ResourceFactoryT.ResourceT
import shapeless.PolyDefns.Case

import scalaz.{Applicative, Monoid, \/, \/-}
import scalaz.concurrent.{Future, Task}

/**
  * A namespace of common operators for Float layers.
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object Float {

  type FloatTape = Tape.Aux[Float, Float]

  @inline
  implicit def toFloatTapeTask[A](
      implicit typeClass: ToRAIITask.Aux[A, Float, Float]): ToRAIITask.Aux[A, Float, Float] = typeClass

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

  object Weight {

    def apply(value: Float)(implicit optimizerFactory: OptimizerFactory) = new Weight(value) {
      override protected val optimizer = optimizerFactory.FloatOptimizer(this)
    }

  }

  // TODO: think about if Weight should be abstract
  abstract case class Weight(var data: Float) extends Tape {
    protected def optimizer: Optimizer

    override type Data = Float
    override type Delta = Float

//    override def value: \/-[this.type] = \/-(this)
//
//    override def acquire(): Future[ResourceT[Future, \/[Throwable, Aux[Float, Float]]]] = Future.now(this)
//
//    override def release(): Future[Unit] = Future.now(())

    override def backward(deltaFuture: Future[Delta]): Future[Unit] = {
      deltaFuture.map { delta =>
        data = optimizer.updateFloat(data, delta)
      }
    }
  }

  implicit final class WeightOps(value: Float) {
    def toWeight(implicit optimizerFactory: OptimizerFactory): Weight = {
      Weight(value)
    }
  }

  implicit final class RAIIOps[Data, Delta](value: Tape.Aux[Data, Delta]) {
    def toRAIITask: RAIITask[Tape.Aux[Data, Delta]] = {
      RAIITask.unmanaged(value)
    }
  }

  object OptimizerFactory {
    implicit def shared(implicit optimizer: Optimizer): OptimizerFactory = new OptimizerFactory {
      override def FloatOptimizer(weight: Weight): Optimizer = optimizer
    }
  }

  trait OptimizerFactory {
    def FloatOptimizer(weight: Weight): Optimizer
  }

  private implicit object FloatMonoid extends Monoid[Float] {
    override def zero: Float = 0.0f

    override def append(f1: Float, f2: => Float): Float = f1 + f2
  }
//
//  implicit final class FloatComputeOps(operand0: Compute[Tape.Aux[Float, Float]]) {
//    def +(operand1: Compute[Tape.Aux[Float, Float]]): Compute[Tape.Aux[Float, Float]] = {
//      Compute.binary(operand0, operand1) { (data0, data1) =>
//        Task.delay {
//          val outputData = data0 + data1
//          def computeDeltas(delta: Float) = {
//            val delta0Future = Future.now(delta)
//            val delta1Future = Future.now(delta)
//            (delta0Future, delta1Future)
//          }
//          (outputData, computeDeltas)
//        }
//      }
//    }
//  }
//
//  final case class Plus(operand0: Compute[Tape.Aux[Float, Float]], operand1: Compute[Tape.Aux[Float, Float]])
//      extends Compute.BinaryComputeFactory[Float, Float] {
//    def apply(operand0: Compute[Aux[Float, Float]], operand1: Compute[Aux[Float, Float]])(
//        computeForward: (Float, Float) => Task[(Float, (Float) => (Future[Float], Future[Float]))])
//      : Compute[Tape.Aux[Float, Float]] = ???
//  }

  implicit val `Float+Float`
    : +.Case.Aux[RAIITask.Covariant[FloatTape], RAIITask.Covariant[FloatTape], RAIITask[FloatTape]] = {
    MathMethods.+.at { (operand0, operand1) =>
      TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
        Task.delay {
          val outputData = data0 + data1
          def computeBackward(delta: Float) = {
            val delta0Future = Future.now(delta)
            val delta1Future = Future.now(delta)
            (delta0Future, delta1Future)
          }
          (outputData, computeBackward)
        }
      }
    }
  }
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
