package com.thoughtworks.deeplearning

import com.thoughtworks.raii.RAIITask
import com.thoughtworks.deeplearning.Float.Optimizers.Optimizer
import com.thoughtworks.deeplearning.PolyFunctions._
import com.thoughtworks.deeplearning.TapeTask.Trainable
import shapeless.the

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
      implicit typeClass: ToTapeTask.Aux[A, Float, Float]): ToTapeTask.Aux[A, Float, Float] = typeClass

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

  final case class Weight(var data: Float)(implicit optimizerFactory: OptimizerFactory) extends Tape {
    private val optimizer: Optimizer = optimizerFactory.FloatOptimizer(this)

    override type Data = Float
    override type Delta = Float

    override def backward(deltaFuture: Future[Delta]): Future[Unit] = {
      deltaFuture.map { delta =>
        synchronized {
          data = optimizer.updateFloat(data, delta)
        }
      }
    }
  }

  implicit final class ToWeightOps(value: Float) {
    def toWeight(implicit optimizerFactory: OptimizerFactory): Weight = {
      Weight(value)
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

  implicit def trainableFloat: Trainable[Float, Float] = new Trainable[Float, Float] {
    override def apply(data: Float): Future[Float] = Future.now(the[Numeric[Float]].one)
  }

  private implicit object FloatMonoid extends Monoid[Float] {
    override def zero: Float = 0.0f

    override def append(f1: Float, f2: => Float): Float = f1 + f2
  }

  @inline
  implicit val `Float+Float`
    : PolyMethods.+.Case.Aux[RAIITask.Covariant[FloatTape], RAIITask.Covariant[FloatTape], RAIITask[FloatTape]] = {
    PolyMethods.+.at { (operand0, operand1) =>
      TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
        Task.delay {
          val outputData = data0 + data1
          val computeBackward = { outputDelta: Float =>
            val delta0Future = Future.now(outputDelta)
            val delta1Future = Future.now(outputDelta)
            (delta0Future, delta1Future)
          }
          (outputData, computeBackward)
        }
      }
    }
  }
}
