package com.thoughtworks.deeplearning
package differentiable

import java.util.logging.{Level, Logger}

import com.thoughtworks.Constructor
import com.thoughtworks.Override.inject
import com.thoughtworks.deeplearning.logs.{UncaughtExceptionDuringBackward, WeightIsUpdating}
import com.thoughtworks.deeplearning.math._
import com.thoughtworks.deeplearning.differentiable.Any.Trainable
import com.thoughtworks.deeplearning.Lift.LowPriorityLift
import com.thoughtworks.raii.asynchronous.Do
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

  trait Hyperparameter {

    trait Weight {

      var data: scala.Float

      final def backward(deltaFuture: Do[scala.Float])(implicit logger: Logger = Logger.getGlobal,
                                                       fullName: sourcecode.FullName,
                                                       className: Caller[_],
                                                       methodName: sourcecode.Name): Future[Unit] = {
        Do.run(Do.releaseMap(deltaFuture) { delta =>
            data -= optimizerConstructor.newInstance(this, delta).delta
          })
          .get
          .map {
            case \/-(()) => ()
            case -\/(e) => logger.log(UncaughtExceptionDuringBackward(e))

          }
      }

    }

    type FloatWeight <: Weight

    type FloatOptimizer <: Optimizer

    abstract class Optimizer(val weight: FloatWeight, delta0: scala.Float) {
      def delta: scala.Float = delta0
    }

    @inject
    def optimizerConstructor: Constructor[(Weight, scala.Float) => FloatOptimizer]

  }

  object Hyperparameter {

    trait FloatInitialization extends Hyperparameter {

      abstract class Weight(override var data: scala.Float) extends super.Weight
      @inject
      def fromFloatConstructor: Constructor[scala.Float => Weight with FloatWeight]

      final def floatWeight(data: scala.Float): Weight with FloatWeight =
        fromFloatConstructor.newInstance(data)

    }

    trait FixedLearningRate extends LearningRate {
      def fixedLearningRate: scala.Float
      trait Optimizer extends super.Optimizer {
        final def learningRate: scala.Float = fixedLearningRate
      }
      override type FloatOptimizer <: Optimizer
    }

    trait LearningRate extends Hyperparameter {
      trait Optimizer extends super.Optimizer {
        def learningRate: scala.Float
        override def delta: scala.Float = super.delta * learningRate
      }
      override type FloatOptimizer <: Optimizer
    }

    trait L1Regularization extends Hyperparameter {
      def l1Regularization: scala.Float
      trait Optimizer extends super.Optimizer {
        override def delta: scala.Float = super.delta + scala.math.signum(weight.data) * l1Regularization
      }
      override type FloatOptimizer <: Optimizer
    }
    trait L2Regularization extends Hyperparameter {
      def l2Regularization: scala.Float
      trait Optimizer extends super.Optimizer {
        override def delta: scala.Float = super.delta + weight.data * l2Regularization
      }
      override type FloatOptimizer <: Optimizer
    }

  }

  object implicits {
    import com.thoughtworks.deeplearning.tapefactories.Binary.monoidBinaryTapeTaskFactory
    import com.thoughtworks.deeplearning.tapefactories.Unary.monoidUnaryTapeTaskFactory

    implicit def liftFloatWeight[W <: Hyperparameter#Weight](implicit logger: Logger = Logger.getGlobal,
                                                             fullName: sourcecode.FullName,
                                                             className: Caller[_],
                                                             methodName: sourcecode.Name) = new Lift[W] {
      override type Data = scala.Float
      override type Delta = scala.Float
      override def apply(weight: W): Do[Tape[Data, Delta]] = {
        import weight._
        Do.delay(Tape(data, backward))
      }
    }
    private[deeplearning] implicit object FloatMonoid extends Monoid[scala.Float] {
      override def zero: scala.Float = 0

      override def append(f1: scala.Float, f2: => scala.Float): scala.Float = f1 + f2
    }

    def infer(self: AnyRef): self.type = self

    @inline
    implicit def liftFloat[A](
        implicit typeClass: LowPriorityLift.Aux[A, scala.Float, scala.Float]): Lift.Aux[A, scala.Float, scala.Float] =
      typeClass

    implicit object FloatTrainable extends Trainable[scala.Float, scala.Float] {
      override def apply(data: scala.Float): Do[scala.Float] = Do.now(1)
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
            val computeBackward = { outputDelta: scala.Float =>
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
            val computeBackward = { outputDelta: scala.Float =>
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
            val computeBackward = { outputDelta: scala.Float =>
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
            val computeBackward = { outputDelta: scala.Float =>
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
            val computeBackward = { outputDelta: scala.Float =>
              val zero = Do.now(the[Numeric[scala.Float]].zero)
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
            val computeBackward = { outputDelta: scala.Float =>
              val zero = Do.now(the[Numeric[scala.Float]].zero)
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
            val computeBackward = { outputDelta: scala.Float =>
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
            val computeBackward = { outputDelta: scala.Float =>
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
            val computeBackward = { outputDelta: scala.Float =>
              if (isDataPositive) Do.now(outputDelta) else Do.delay(-outputDelta)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    def reciprocal[Operand](operand: Operand)(implicit liftOperand: Lift.Aux[Operand, scala.Float, scala.Float],
                                              logger: Logger = Logger.getGlobal,
                                              fullName: sourcecode.FullName,
                                              methodName: sourcecode.Name,
                                              className: Caller[_]): Do[FloatTape] = {
      tapefactories.Unary.doTape(liftOperand(operand)) { data: scala.Float =>
        Task.delay {
          val outputData = 1 / data
          val computeBackward = { outputDelta: scala.Float =>
            Do.delay(-outputDelta / (data * data))
          }
          (outputData, computeBackward)
        }
      }
    }

    implicit final class DifferentiableFloatOps[From](from: From)(
        implicit lift: Lift.Aux[From, scala.Float, scala.Float],
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
            val computeBackward = { outputDelta: scala.Float =>
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
private[differentiable] abstract class FloatCompanion { this: Float.type =>
  private[deeplearning] type FloatTape = Tape[scala.Float, scala.Float]
}
