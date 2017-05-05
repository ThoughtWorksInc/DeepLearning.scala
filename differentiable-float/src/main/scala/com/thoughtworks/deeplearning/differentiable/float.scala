package com.thoughtworks.deeplearning
package differentiable

import java.util.logging.{Level, Logger}

import com.thoughtworks.deeplearning.LogRecords.{UncaughtExceptionDuringBackward, WeightIsUpdating}
import com.thoughtworks.deeplearning.PolyFunctions._
import com.thoughtworks.deeplearning.TapeTask.Trainable
import com.thoughtworks.deeplearning.ToTapeTask.LowPriorityToTapeTask
import com.thoughtworks.raii.future.Do
import com.thoughtworks.raii.ownership._
import com.thoughtworks.raii.ownership.implicits._
import com.thoughtworks.raii.transformers.ResourceFactoryT
import shapeless.the

import scala.util.{Failure, Success, Try}
import scalaz.{-\/, Applicative, Monoid, \/, \/-}
import scalaz.concurrent.{Future, Task}
import scalaz.syntax.all._

/**
  * A namespace of common operators for Float layers.
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object float {

  private[deeplearning] type FloatTape = Borrowing[Tape.Aux[Float, Float]]

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
                                           logger: Logger = Logger.getGlobal,
                                           fullName: sourcecode.FullName,
                                           className: Caller[_],
                                           methodName: sourcecode.Name)
      extends Tape {
    private val optimizer: Optimizer = optimizerFactory.floatOptimizer(this)

    override type Data = Float
    override type Delta = Float

    override def backward(deltaFuture: Do[_ <: Delta]): Future[Unit] = {
      import com.thoughtworks.raii.transformers.ResourceFactoryT.resourceFactoryTMonad

      val Do(resourceFactoryTFuture) = deltaFuture

      val resourceFactoryT: ResourceFactoryT[Future, Try[Delta]] = ResourceFactoryT(resourceFactoryTFuture)

      val tryTRAIIFuture: ResourceFactoryT[Future, Try[Unit]] = resourceFactoryT.map { tryDelta: Try[Delta] =>
        tryDelta.map { delta =>
          synchronized {
            if (logger.isLoggable(Level.FINER)) {
              logger.log(WeightIsUpdating(data, delta))
            }
            data = optimizer.updateFloat(data, delta)
          }
        }
      }

      ResourceFactoryT.run(tryTRAIIFuture).flatMap {
        case Failure(e) =>
          logger.log(UncaughtExceptionDuringBackward(e))
          Future.now(())
        case Success(()) =>
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
      override def zero: Float = 0

      override def append(f1: Float, f2: => Float): Float = f1 + f2
    }

    def infer(self: AnyRef): self.type = self

    @inline
    implicit def liftFloat[A](
        implicit typeClass: LowPriorityToTapeTask.Aux[A, Float, Float]): ToTapeTask.Aux[A, Float, Float] =
      typeClass

    implicit final class FloatToWeightOps(value: Float) {
      def toWeight(implicit optimizerFactory: OptimizerFactory,
                   logger: Logger = Logger.getGlobal,
                   fullName: sourcecode.FullName,
                   className: Caller[_],
                   methodName: sourcecode.Name): Do[Borrowing[Tape.Aux[Float, Float]]] = {
        val myWeight = garbageCollectable(Weight(value))
        Do.now(myWeight)
      }
    }

    implicit object FloatTrainable extends Trainable[Float, Float] {
      override def apply(data: Float): Do[Float] = Do.now(1)
    }

    @inline
    implicit def `Float+Float`(implicit logger: Logger = Logger.getGlobal,
                               fullName: sourcecode.FullName,
                               className: Caller[_],
                               methodName: sourcecode.Name)
      : PolyMethods.+.Case.Aux[Do.Covariant[FloatTape], Do.Covariant[FloatTape], Do[FloatTape]] = {
      PolyMethods.+.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val outputData = data0 + data1
            val computeBackward = { outputDelta: Float =>
              val delta0Future = Do.now(outputDelta)
              val delta1Future = Do.now(outputDelta)
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `Float-Float`(implicit logger: Logger = Logger.getGlobal,
                               fullName: sourcecode.FullName,
                               className: Caller[_],
                               methodName: sourcecode.Name)
      : PolyMethods.-.Case.Aux[Do.Covariant[FloatTape], Do.Covariant[FloatTape], Do[FloatTape]] = {
      PolyMethods.-.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val outputData = data0 - data1
            val computeBackward = { outputDelta: Float =>
              val delta0Future = Do.now(outputDelta)
              val delta1Future = Do.delay(-outputDelta)
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `Float*Float`(implicit logger: Logger = Logger.getGlobal,
                               fullName: sourcecode.FullName,
                               className: Caller[_],
                               methodName: sourcecode.Name)
      : PolyMethods.*.Case.Aux[Do.Covariant[FloatTape], Do.Covariant[FloatTape], Do[FloatTape]] = {
      PolyMethods.*.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val outputData = data0 * data1
            val computeBackward = { outputDelta: Float =>
              val delta0Future = Do.delay(outputDelta * data1)
              val delta1Future = Do.delay(outputDelta * data0)
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `Float/Float`(implicit logger: Logger = Logger.getGlobal,
                               fullName: sourcecode.FullName,
                               className: Caller[_],
                               methodName: sourcecode.Name)
      : PolyMethods./.Case.Aux[Do.Covariant[FloatTape], Do.Covariant[FloatTape], Do[FloatTape]] = {
      PolyMethods./.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val outputData = data0 / data1
            val computeBackward = { outputDelta: Float =>
              val delta0Future = Do.delay(outputDelta / data1)
              val delta1Future = Do.delay(-data0 * outputDelta / (data1 * data1))
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `min(Float,Float)`(implicit logger: Logger = Logger.getGlobal,
                                    fullName: sourcecode.FullName,
                                    className: Caller[_],
                                    methodName: sourcecode.Name)
      : PolyFunctions.min.Case.Aux[Do.Covariant[FloatTape], Do.Covariant[FloatTape], Do[FloatTape]] = {
      PolyFunctions.min.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val leftLessThenRight = data0 < data1
            val outputData = if (leftLessThenRight) data0 else data1
            val computeBackward = { outputDelta: Float =>
              val zero = Do.now(the[Numeric[Float]].zero)
              val delta = Do.now(outputDelta)
              if (leftLessThenRight) (delta, zero) else (zero, delta)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `max(Float,Float)`(implicit logger: Logger = Logger.getGlobal,
                                    fullName: sourcecode.FullName,
                                    className: Caller[_],
                                    methodName: sourcecode.Name)
      : PolyFunctions.max.Case.Aux[Do.Covariant[FloatTape], Do.Covariant[FloatTape], Do[FloatTape]] = {
      PolyFunctions.max.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val leftLessThenRight = data0 < data1
            val outputData = if (leftLessThenRight) data1 else data0
            val computeBackward = { outputDelta: Float =>
              val zero = Do.now(the[Numeric[Float]].zero)
              val delta = Do.now(outputDelta)
              if (leftLessThenRight) (zero, delta) else (delta, zero)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `log(Float)`(
        implicit logger: Logger = Logger.getGlobal,
        fullName: sourcecode.FullName,
        className: Caller[_],
        methodName: sourcecode.Name): PolyFunctions.log.Case.Aux[Do.Covariant[FloatTape], Do[FloatTape]] = {
      PolyFunctions.log.at { operand =>
        TapeTaskFactory.unary(operand) { data =>
          Task.delay {
            val outputData = math.log(data).toFloat
            val computeBackward = { outputDelta: Float =>
              Do.delay(outputDelta / data)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `exp(Float)`(
        implicit logger: Logger = Logger.getGlobal,
        fullName: sourcecode.FullName,
        methodName: sourcecode.Name,
        className: Caller[_]): PolyFunctions.exp.Case.Aux[Do.Covariant[FloatTape], Do[FloatTape]] = {
      PolyFunctions.exp.at { operand =>
        TapeTaskFactory.unary(operand) { data =>
          Task.delay {
            val outputData = math.exp(data).toFloat
            val computeBackward = { outputDelta: Float =>
              Do.delay(outputDelta * outputData)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `abs(Float)`(
        implicit logger: Logger = Logger.getGlobal,
        fullName: sourcecode.FullName,
        methodName: sourcecode.Name,
        className: Caller[_]): PolyFunctions.abs.Case.Aux[Do.Covariant[FloatTape], Do[FloatTape]] = {
      PolyFunctions.abs.at { operand =>
        TapeTaskFactory.unary(operand) { data =>
          Task.delay {
            val isDataPositive = data >= 0
            val outputData = if (isDataPositive) data else -data
            val computeBackward = { outputDelta: Float =>
              if (isDataPositive) Do.now(outputDelta) else Do.delay(-outputDelta)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    implicit final class DifferentiableFloatOps[From](from: From)(implicit lift: ToTapeTask.Aux[From, Float, Float],
                                                                  logger: Logger = Logger.getGlobal,
                                                                  fullName: sourcecode.FullName,
                                                                  methodName: sourcecode.Name,
                                                                  className: Caller[_]) {
      private val operand: Do.Covariant[FloatTape] = lift(from)
      @inline
      def unary_- : Do[FloatTape] = {
        TapeTaskFactory.unary(operand) { data =>
          Task.delay {
            val outputData = -data
            val computeBackward = { outputDelta: Float =>
              Do.delay(-outputDelta)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

  }
}
