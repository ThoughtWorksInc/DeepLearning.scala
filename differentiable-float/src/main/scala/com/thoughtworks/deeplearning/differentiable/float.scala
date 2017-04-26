package com.thoughtworks.deeplearning
package differentiable

import java.util.logging.{Level, Logger}

import com.thoughtworks.raii.RAIITask
import com.thoughtworks.deeplearning.LogRecords.{FloatWeightTracker, UncaughtExceptionDuringBackward}
import com.thoughtworks.deeplearning.PolyFunctions._
import com.thoughtworks.deeplearning.TapeTask.Trainable
import shapeless.the

import scalaz.{-\/, Applicative, Monoid, \/, \/-}
import scalaz.concurrent.{Future, Task}

/**
  * A namespace of common operators for Float layers.
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object float {

  private[deeplearning] type FloatTape = Tape.Aux[Float, Float]

  trait Optimizer {
    def currentDelta(oldValue: Float, delta: Float): Float = delta

    final def updateFloat(oldValue: Float, delta: Float): Float = {
      oldValue - currentDelta(oldValue, delta)
    }
  }

  object Optimizer {

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

  final case class Weight(var data: Float)(implicit optimizerFactory: OptimizerFactory,
                                           logger: Logger = Logger.getGlobal)
      extends Tape {
    private val optimizer: Optimizer = optimizerFactory.floatOptimizer(this)

    override type Data = Float
    override type Delta = Float

    override def backward(deltaFuture: RAIITask[_ <: Delta]): Future[Unit] = {
      val eitherTRAIIFuture = deltaFuture.map { delta =>
        synchronized {
          if (logger.isLoggable(Level.FINER)) {
            logger.log(FloatWeightTracker(s"weight: $data, delta:$delta"))
          }
          data = optimizer.updateFloat(data, delta)
        }
      }

      eitherTRAIIFuture.run.run.flatMap {
        case -\/(e) =>
          logger.log(UncaughtExceptionDuringBackward(e, "An exception raised during backward"))
          Future.now(())
        case \/-(()) =>
          Future.now(())
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

  object implicits {

    private implicit object FloatMonoid extends Monoid[Float] {
      override def zero: Float = the[Numeric[Float]].zero

      override def append(f1: Float, f2: => Float): Float = f1 + f2
    }

    @inline
    implicit def liftFloat[A](implicit typeClass: ToTapeTask.Aux[A, Float, Float]): ToTapeTask.Aux[A, Float, Float] =
      typeClass
    implicit final class FloatToWeightOps(value: Float) {
      def toWeight(implicit optimizerFactory: OptimizerFactory, logger: Logger = Logger.getGlobal): Weight = {
        Weight(value)
      }
    }

    implicit object FloatTrainable extends Trainable[Float, Float] {
      override def apply(data: Float): RAIITask[Float] = RAIITask.now(the[Numeric[Float]].one)
    }

    @inline
    implicit def `Float+Float`(implicit logger: Logger = Logger.getGlobal)
      : PolyMethods.+.Case.Aux[RAIITask.Covariant[FloatTape], RAIITask.Covariant[FloatTape], RAIITask[FloatTape]] = {
      PolyMethods.+.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val outputData = data0 + data1
            val computeBackward = { outputDelta: Float =>
              val delta0Future = RAIITask.now(outputDelta)
              val delta1Future = RAIITask.now(outputDelta)
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `Float-Float`(implicit logger: Logger = Logger.getGlobal)
      : PolyMethods.-.Case.Aux[RAIITask.Covariant[FloatTape], RAIITask.Covariant[FloatTape], RAIITask[FloatTape]] = {
      PolyMethods.-.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val outputData = data0 - data1
            val computeBackward = { outputDelta: Float =>
              val delta0Future = RAIITask.now(outputDelta)
              val delta1Future = RAIITask.delay(-outputDelta)
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `Float*Float`(implicit logger: Logger = Logger.getGlobal)
      : PolyMethods.*.Case.Aux[RAIITask.Covariant[FloatTape], RAIITask.Covariant[FloatTape], RAIITask[FloatTape]] = {
      PolyMethods.*.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val outputData = data0 * data1
            val computeBackward = { outputDelta: Float =>
              val delta0Future = RAIITask.delay(outputDelta * data1)
              val delta1Future = RAIITask.delay(outputDelta * data0)
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `Float/Float`(implicit logger: Logger = Logger.getGlobal)
      : PolyMethods./.Case.Aux[RAIITask.Covariant[FloatTape], RAIITask.Covariant[FloatTape], RAIITask[FloatTape]] = {
      PolyMethods./.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val outputData = data0 / data1
            val computeBackward = { outputDelta: Float =>
              val delta0Future = RAIITask.delay(outputDelta / data1)
              val delta1Future = RAIITask.delay(-data0 * outputDelta / (data1 * data1))
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `min(Float,Float)`(
        implicit logger: Logger = Logger.getGlobal): PolyFunctions.min.Case.Aux[RAIITask.Covariant[FloatTape],
                                                                                RAIITask.Covariant[FloatTape],
                                                                                RAIITask[FloatTape]] = {
      PolyFunctions.min.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val leftLessThenRight = data0 < data1
            val outputData = if (leftLessThenRight) data0 else data1
            val computeBackward = { outputDelta: Float =>
              val zero = RAIITask.now(the[Numeric[Float]].zero)
              val delta = RAIITask.now(outputDelta)
              if (leftLessThenRight) (delta, zero) else (zero, delta)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `max(Float,Float)`(
        implicit logger: Logger = Logger.getGlobal): PolyFunctions.max.Case.Aux[RAIITask.Covariant[FloatTape],
                                                                                RAIITask.Covariant[FloatTape],
                                                                                RAIITask[FloatTape]] = {
      PolyFunctions.max.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val leftLessThenRight = data0 < data1
            val outputData = if (leftLessThenRight) data1 else data0
            val computeBackward = { outputDelta: Float =>
              val zero = RAIITask.now(the[Numeric[Float]].zero)
              val delta = RAIITask.now(outputDelta)
              if (leftLessThenRight) (zero, delta) else (delta, zero)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `log(Float)`(implicit logger: Logger = Logger.getGlobal)
      : PolyFunctions.log.Case.Aux[RAIITask.Covariant[FloatTape], RAIITask[FloatTape]] = {
      PolyFunctions.log.at { operand =>
        TapeTaskFactory.unary(operand) { data =>
          Task.delay {
            val outputData = math.log(data).toFloat
            val computeBackward = { outputDelta: Float =>
              RAIITask.delay(outputDelta / data)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `exp(Float)`(implicit logger: Logger = Logger.getGlobal)
      : PolyFunctions.exp.Case.Aux[RAIITask.Covariant[FloatTape], RAIITask[FloatTape]] = {
      PolyFunctions.exp.at { operand =>
        TapeTaskFactory.unary(operand) { data =>
          Task.delay {
            val outputData = math.exp(data).toFloat
            val computeBackward = { outputDelta: Float =>
              RAIITask.delay(outputDelta * outputData)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `abs(Float)`(implicit logger: Logger = Logger.getGlobal)
      : PolyFunctions.abs.Case.Aux[RAIITask.Covariant[FloatTape], RAIITask[FloatTape]] = {
      PolyFunctions.abs.at { operand =>
        TapeTaskFactory.unary(operand) { data =>
          Task.delay {
            val isDataPositive = data >= 0
            val outputData = if (isDataPositive) data else -data
            val computeBackward = { outputDelta: Float =>
              if (isDataPositive) RAIITask.now(outputDelta) else RAIITask.delay(-outputDelta)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    implicit final class DifferentiableFloatOps[From](from: From)(implicit lift: ToTapeTask.Aux[From, Float, Float],
                                                    logger: Logger = Logger.getGlobal) {
      private val operand: RAIITask.Covariant[FloatTape] = lift(from)
      @inline
      def unary_- : RAIITask[FloatTape] = {
        TapeTaskFactory.unary(operand) { data =>
          Task.delay {
            val outputData = -data
            val computeBackward = { outputDelta: Float =>
              RAIITask.delay(-outputDelta)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

  }
}
