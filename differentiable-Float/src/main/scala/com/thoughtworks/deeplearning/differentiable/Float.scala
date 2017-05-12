package com.thoughtworks.deeplearning
package differentiable

import scala.{Float => ScalaFloat}

import java.util.logging.{Level, Logger}

import com.thoughtworks.deeplearning.logs.{UncaughtExceptionDuringBackward, WeightIsUpdating}
import com.thoughtworks.deeplearning.math._
import com.thoughtworks.deeplearning.differentiable.Any.Trainable
import com.thoughtworks.deeplearning.Lift.LowPriorityLift
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous._
import com.thoughtworks.raii.ownership._
import com.thoughtworks.raii.ownership._
import com.thoughtworks.raii.covariant.ResourceT
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
object Float extends FloatCompanion {

  trait Optimizer {
    def currentDelta(oldValue: ScalaFloat, delta: ScalaFloat): ScalaFloat = delta

    final def updateFloat(oldValue: ScalaFloat, delta: ScalaFloat): ScalaFloat = {
      oldValue - currentDelta(oldValue, delta)
    }
  }

  object Optimizer {

    trait LearningRate extends Optimizer {

      protected def currentLearningRate(): ScalaFloat

      override def currentDelta(oldValue: ScalaFloat, delta: ScalaFloat): ScalaFloat = delta * currentLearningRate()
    }

    trait L1Regularization extends Optimizer {
      protected def l1Regularization: ScalaFloat

      override def currentDelta(oldValue: ScalaFloat, delta: ScalaFloat): ScalaFloat = {
        super.currentDelta(oldValue, delta + scala.math.signum(oldValue) * l1Regularization)
      }
    }

    trait L2Regularization extends Optimizer {
      protected def l2Regularization: ScalaFloat

      override def currentDelta(oldValue: ScalaFloat, delta: ScalaFloat): ScalaFloat = {
        super.currentDelta(oldValue, delta + oldValue * l2Regularization)
      }
    }

  }

  final case class Weight(var data: ScalaFloat)(implicit optimizerFactory: OptimizerFactory,
                                                logger: Logger = Logger.getGlobal,
                                                fullName: sourcecode.FullName,
                                                className: Caller[_],
                                                methodName: sourcecode.Name) {

    def doTape = Do.delay(garbageCollectable(Tape(data, backward)))
    private val optimizer: Optimizer = optimizerFactory.floatOptimizer(this)

    def backward(deltaFuture: Do[ScalaFloat]): Future[Unit] = {
      import com.thoughtworks.raii.covariant.ResourceT.resourceTMonad

      val Do(resourceTFuture) = deltaFuture

      val resourceT: ResourceT[Future, Try[ScalaFloat]] = ResourceT(resourceTFuture)

      val tryTRAIIFuture: ResourceT[Future, Try[Unit]] = resourceT.map { tryScalaFloat: Try[ScalaFloat] =>
        tryScalaFloat.map { delta =>
          synchronized {
            if (logger.isLoggable(Level.FINER)) {
              logger.log(WeightIsUpdating(data, delta))
            }
            data = optimizer.updateFloat(data, delta)
          }
        }
      }

      ResourceT.run(tryTRAIIFuture).flatMap {
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
    import com.thoughtworks.deeplearning.tapefactories.Binary.monoidBinaryTapeTaskFactory
    import com.thoughtworks.deeplearning.tapefactories.Unary.monoidUnaryTapeTaskFactory

    private[deeplearning] implicit object FloatMonoid extends Monoid[ScalaFloat] {
      override def zero: ScalaFloat = 0

      override def append(f1: ScalaFloat, f2: => ScalaFloat): ScalaFloat = f1 + f2
    }

    def infer(self: AnyRef): self.type = self

    @inline
    implicit def liftFloat[A](
        implicit typeClass: LowPriorityLift.Aux[A, ScalaFloat, ScalaFloat]): Lift.Aux[A, ScalaFloat, ScalaFloat] =
      typeClass

    implicit final class FloatToWeightOps(value: ScalaFloat) {
      def toWeight(implicit optimizerFactory: OptimizerFactory,
                   logger: Logger = Logger.getGlobal,
                   fullName: sourcecode.FullName,
                   className: Caller[_],
                   methodName: sourcecode.Name): Do[Borrowing[Tape[ScalaFloat, ScalaFloat]]] = {
        Weight(value).doTape
      }
    }

    implicit object FloatTrainable extends Trainable[ScalaFloat, ScalaFloat] {
      override def apply(data: ScalaFloat): Do[ScalaFloat] = Do.now(1)
    }

    @inline
    implicit def `Float+Float`(
        implicit logger: Logger = Logger.getGlobal,
        fullName: sourcecode.FullName,
        className: Caller[_],
        methodName: sourcecode.Name): polyFunctions.+.Case.Aux[Do[FloatTape], Do[FloatTape], Do[FloatTape]] = {
      polyFunctions.+.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val outputData = data0 + data1
            val computeBackward = { outputDelta: ScalaFloat =>
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
    implicit def `Float-Float`(
        implicit logger: Logger = Logger.getGlobal,
        fullName: sourcecode.FullName,
        className: Caller[_],
        methodName: sourcecode.Name): polyFunctions.-.Case.Aux[Do[FloatTape], Do[FloatTape], Do[FloatTape]] = {
      polyFunctions.-.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val outputData = data0 - data1
            val computeBackward = { outputDelta: ScalaFloat =>
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
    implicit def `Float*Float`(
        implicit logger: Logger = Logger.getGlobal,
        fullName: sourcecode.FullName,
        className: Caller[_],
        methodName: sourcecode.Name): polyFunctions.*.Case.Aux[Do[FloatTape], Do[FloatTape], Do[FloatTape]] = {
      polyFunctions.*.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val outputData = data0 * data1
            val computeBackward = { outputDelta: ScalaFloat =>
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
    implicit def `Float/Float`(
        implicit logger: Logger = Logger.getGlobal,
        fullName: sourcecode.FullName,
        className: Caller[_],
        methodName: sourcecode.Name): polyFunctions./.Case.Aux[Do[FloatTape], Do[FloatTape], Do[FloatTape]] = {
      polyFunctions./.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val outputData = data0 / data1
            val computeBackward = { outputDelta: ScalaFloat =>
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
    implicit def `min(Float,Float)`(
        implicit logger: Logger = Logger.getGlobal,
        fullName: sourcecode.FullName,
        className: Caller[_],
        methodName: sourcecode.Name): math.polyFunctions.min.Case.Aux[Do[FloatTape], Do[FloatTape], Do[FloatTape]] = {
      math.polyFunctions.min.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val leftLessThenRight = data0 < data1
            val outputData = if (leftLessThenRight) data0 else data1
            val computeBackward = { outputDelta: ScalaFloat =>
              val zero = Do.now(the[Numeric[ScalaFloat]].zero)
              val delta = Do.now(outputDelta)
              if (leftLessThenRight) (delta, zero) else (zero, delta)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `max(Float,Float)`(
        implicit logger: Logger = Logger.getGlobal,
        fullName: sourcecode.FullName,
        className: Caller[_],
        methodName: sourcecode.Name): math.polyFunctions.max.Case.Aux[Do[FloatTape], Do[FloatTape], Do[FloatTape]] = {
      math.polyFunctions.max.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val leftLessThenRight = data0 < data1
            val outputData = if (leftLessThenRight) data1 else data0
            val computeBackward = { outputDelta: ScalaFloat =>
              val zero = Do.now(the[Numeric[ScalaFloat]].zero)
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
        methodName: sourcecode.Name): math.polyFunctions.log.Case.Aux[Do[FloatTape], Do[FloatTape]] = {
      math.polyFunctions.log.at { operand =>
        tapefactories.Unary.doTape(operand) { data =>
          Task.delay {
            val outputData = scala.math.log(data).toFloat
            val computeBackward = { outputDelta: ScalaFloat =>
              Do.delay(outputDelta / data)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `exp(Float)`(implicit logger: Logger = Logger.getGlobal,
                              fullName: sourcecode.FullName,
                              methodName: sourcecode.Name,
                              className: Caller[_]): math.polyFunctions.exp.Case.Aux[Do[FloatTape], Do[FloatTape]] = {
      math.polyFunctions.exp.at { operand =>
        tapefactories.Unary.doTape(operand) { data =>
          Task.delay {
            val outputData = scala.math.exp(data).toFloat
            val computeBackward = { outputDelta: ScalaFloat =>
              Do.delay(outputDelta * outputData)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `abs(Float)`(implicit logger: Logger = Logger.getGlobal,
                              fullName: sourcecode.FullName,
                              methodName: sourcecode.Name,
                              className: Caller[_]): math.polyFunctions.abs.Case.Aux[Do[FloatTape], Do[FloatTape]] = {
      math.polyFunctions.abs.at { operand =>
        tapefactories.Unary.doTape(operand) { data =>
          Task.delay {
            val isDataPositive = data >= 0
            val outputData = if (isDataPositive) data else -data
            val computeBackward = { outputDelta: ScalaFloat =>
              if (isDataPositive) Do.now(outputDelta) else Do.delay(-outputDelta)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    def reciprocal[Operand](operand: Operand)(implicit liftOperand: Lift.Aux[Operand, ScalaFloat, ScalaFloat],
                                              logger: Logger = Logger.getGlobal,
                                              fullName: sourcecode.FullName,
                                              methodName: sourcecode.Name,
                                              className: Caller[_]): Do[FloatTape] = {
      tapefactories.Unary.doTape(liftOperand(operand)) { data: ScalaFloat =>
        Task.delay {
          val outputData = 1 / data
          val computeBackward = { outputDelta: ScalaFloat =>
            Do.delay(-outputDelta / (data * data))
          }
          (outputData, computeBackward)
        }
      }
    }

    implicit final class DifferentiableFloatOps[From](from: From)(
        implicit lift: Lift.Aux[From, ScalaFloat, ScalaFloat],
        logger: Logger = Logger.getGlobal,
        fullName: sourcecode.FullName,
        methodName: sourcecode.Name,
        className: Caller[_]) {
      private val operand: Do[FloatTape] = lift(from)
      @inline
      def unary_- : Do[FloatTape] = {
        tapefactories.Unary.doTape(operand) { data =>
          Task.delay {
            val outputData = -data
            val computeBackward = { outputDelta: ScalaFloat =>
              Do.delay(-outputDelta)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

  }
}

//workaround for https://github.com/scala/bug/issues/10306
abstract class FloatCompanion {
  private[deeplearning] type FloatTape = Borrowing[Tape[ScalaFloat, ScalaFloat]]
}
