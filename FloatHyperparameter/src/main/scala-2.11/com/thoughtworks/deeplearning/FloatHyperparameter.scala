package com.thoughtworks.deeplearning

import java.util.logging.Logger

import com.thoughtworks.deeplearning.Lift.LowPriorityLift
import com.thoughtworks.deeplearning.logs.UncaughtExceptionDuringBackward
import com.thoughtworks.feature.{Caller, Factory, ImplicitApply, PartialApply}
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.raii.asynchronous.Do
import shapeless.Witness

import scala.annotation.meta.getter
import scala.concurrent.ExecutionContext
import scalaz.concurrent.Future
import scalaz.{-\/, \/-}

import java.util.logging.{Level, Logger}

import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.deeplearning.logs.{UncaughtExceptionDuringBackward, WeightIsUpdating}
import com.thoughtworks.deeplearning.math._
import com.thoughtworks.deeplearning.Lift.LowPriorityLift
import com.thoughtworks.feature.Caller
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.covariant.ResourceT
import shapeless.the

import scala.util.{Failure, Success, Try}
import scalaz.{-\/, Applicative, Monoid, \/, \/-}
import scalaz.concurrent.{Future, Task}
import scalaz.syntax.all._

/** The base type for all hyperparameters related to [[scala.Float]].
  *
  * This [[FloatHyperparameter]] is usually used with [[com.thoughtworks.feature.Factory Factory]]:
  *
  * {{{
  * import com.thoughtworks.feature.Factory
  * }}}
  *
  * @example Build a [[FloatHyperparameter]] from [[com.thoughtworks.feature.Factory Factory]]
  *          {{{
  *          val differentiable = Factory[FloatHyperparameter].newInstance(logger=java.util.logging.Logger.getGlobal)
  *          val weight: differentiable.FloatWeight = differentiable.FloatWeight(1.3f)
  *
  *          }}}
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait FloatHyperparameter extends Hyperparameter {
  protected implicit object FloatMonoid extends Monoid[scala.Float] {
    override def zero: scala.Float = 0

    override def append(f1: scala.Float, f2: => scala.Float): scala.Float = f1 + f2
  }
  @inline
  protected def floatReciprocal[Operand](operand: Operand)(
      implicit liftOperand: Lift.Aux[Operand, scala.Float, scala.Float],
      fullName: sourcecode.FullName,
      methodName: sourcecode.Name,
      className: Caller[_]): Do[Tape[scala.Float, scala.Float]] = {
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

  trait ImplicitsApi {

    @inline
    implicit def liftFloat[A](implicit neuralNetwork: LowPriorityLift.Aux[A, scala.Float, scala.Float])
      : Lift.Aux[A, scala.Float, scala.Float] =
      neuralNetwork

    @inline
    implicit def liftFloatWeight[SubtypeOfOptimizer](
        implicit implicitApplyRest: ImplicitApply.Aux[floatPartialApplyOriginalDelta.Rest, SubtypeOfOptimizer],
        constraint: SubtypeOfOptimizer <:< FloatOptimizer): Lift.Aux[FloatWeight, scala.Float, scala.Float] = {
      new Lift[FloatWeight] {
        override type Data = scala.Float
        override type Delta = scala.Float

        @inline
        override def apply(weight: FloatWeight): Do[Tape[scala.Float, scala.Float]] = {
          import weight._
          Do.now(Tape(data, backward[SubtypeOfOptimizer]))
        }
      }
    }

    implicit final class DifferentiableFloatOps[From](from: From)(
        implicit lift: Lift.Aux[From, scala.Float, scala.Float],
        fullName: sourcecode.FullName,
        methodName: sourcecode.Name,
        className: Caller[_]) {
      private val operand: Do[Tape[scala.Float, scala.Float]] = lift(from)
      @inline
      def unary_- : Do[Tape[scala.Float, scala.Float]] = {
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

    implicit object FloatLoss extends Loss[scala.Float, scala.Float] {
      override def deltaLoss(data: scala.Float): Do[scala.Float] = Do.now(1)
    }

    @inline
    implicit def `Float+Float`(
        implicit fullName: sourcecode.FullName,
        className: Caller[_],
        methodName: sourcecode.Name): polyFunctions.+.Case.Aux[Do[Tape[scala.Float, scala.Float]],
                                                               Do[Tape[scala.Float, scala.Float]],
                                                               Do[Tape[scala.Float, scala.Float]]] = {
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
        implicit fullName: sourcecode.FullName,
        className: Caller[_],
        methodName: sourcecode.Name): polyFunctions.-.Case.Aux[Do[Tape[scala.Float, scala.Float]],
                                                               Do[Tape[scala.Float, scala.Float]],
                                                               Do[Tape[scala.Float, scala.Float]]] = {
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
        implicit fullName: sourcecode.FullName,
        className: Caller[_],
        methodName: sourcecode.Name): polyFunctions.*.Case.Aux[Do[Tape[scala.Float, scala.Float]],
                                                               Do[Tape[scala.Float, scala.Float]],
                                                               Do[Tape[scala.Float, scala.Float]]] = {
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
        implicit fullName: sourcecode.FullName,
        className: Caller[_],
        methodName: sourcecode.Name): polyFunctions./.Case.Aux[Do[Tape[scala.Float, scala.Float]],
                                                               Do[Tape[scala.Float, scala.Float]],
                                                               Do[Tape[scala.Float, scala.Float]]] = {
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
        implicit fullName: sourcecode.FullName,
        className: Caller[_],
        methodName: sourcecode.Name): math.polyFunctions.min.Case.Aux[Do[Tape[scala.Float, scala.Float]],
                                                                      Do[Tape[scala.Float, scala.Float]],
                                                                      Do[Tape[scala.Float, scala.Float]]] = {
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
        implicit fullName: sourcecode.FullName,
        className: Caller[_],
        methodName: sourcecode.Name): math.polyFunctions.max.Case.Aux[Do[Tape[scala.Float, scala.Float]],
                                                                      Do[Tape[scala.Float, scala.Float]],
                                                                      Do[Tape[scala.Float, scala.Float]]] = {
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
    implicit def `log(Float)`(implicit fullName: sourcecode.FullName,
                              className: Caller[_],
                              methodName: sourcecode.Name)
      : math.polyFunctions.log.Case.Aux[Do[Tape[scala.Float, scala.Float]], Do[Tape[scala.Float, scala.Float]]] = {
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
    implicit def `exp(Float)`(implicit fullName: sourcecode.FullName,
                              methodName: sourcecode.Name,
                              className: Caller[_])
      : math.polyFunctions.exp.Case.Aux[Do[Tape[scala.Float, scala.Float]], Do[Tape[scala.Float, scala.Float]]] = {
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
    implicit def `abs(Float)`(implicit fullName: sourcecode.FullName,
                              methodName: sourcecode.Name,
                              className: Caller[_])
      : math.polyFunctions.abs.Case.Aux[Do[Tape[scala.Float, scala.Float]], Do[Tape[scala.Float, scala.Float]]] = {
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

  }

  type FloatLayer = Do[Tape[scala.Float, scala.Float]]

  type Implicits <: ImplicitsApi

  trait FloatWeightApi { this: FloatWeight =>

    implicit protected val fullName: sourcecode.FullName
    implicit protected val name: sourcecode.Name
    implicit protected val caller: Caller[_]

    var data: scala.Float

    private def optimizer[SubtypeOfOptimizer](delta: scala.Float)(
        implicit implicitApplyRest: ImplicitApply.Aux[floatPartialApplyOriginalDelta.Rest, SubtypeOfOptimizer],
        constraint: SubtypeOfOptimizer <:< FloatOptimizer
    ): FloatOptimizer = {
      constraint(
        implicitApplyRest(
          floatPartialApplyOriginalDelta(floatPartialApplyWeight(floatOptimizerFactory.newInstance,
                                                                 floatWeightParameter(this)),
                                         floatOriginalDeltaParameter(delta))))
    }

    private[FloatHyperparameter] final def backward[SubtypeOfOptimizer](deltaFuture: Do[scala.Float])(
        implicit implicitApplyRest: ImplicitApply.Aux[floatPartialApplyOriginalDelta.Rest, SubtypeOfOptimizer],
        constraint: SubtypeOfOptimizer <:< FloatOptimizer): Future[Unit] = {
      val doUpdate: Do[Unit] = Do.releaseMap(deltaFuture) { delta =>
        data -= optimizer(delta).delta
      }
      Do.run(doUpdate).get.map {
        case \/-(()) => ()
        case -\/(e) =>
          val logRecord = UncaughtExceptionDuringBackward(e)
          logger.log(logRecord)
      }

    }

  }

  type FloatWeight <: FloatWeightApi

  object FloatWeight {
    def apply[SubtypeOfWeight, OptimizerFunction, Optimizer](data: scala.Float)(
        implicit implicitApplyRest: ImplicitApply.Aux[floatPartialApplyData.Rest, SubtypeOfWeight],
        isWeight: SubtypeOfWeight <:< FloatWeight
    ): FloatWeight = {
      implicitApplyRest(floatPartialApplyData(floatWeightFactory.newInstance, floatDataParameter(data)))
    }
  }

  trait FloatOptimizerApi {
    val weight: FloatWeight

    val originalDelta: scala.Float
    def delta: scala.Float = originalDelta
  }
  type FloatOptimizer <: FloatOptimizerApi

  @(inject @getter)
  protected val floatOptimizerFactory: Factory[FloatOptimizer]

  @(inject @getter)
  protected val floatPartialApplyWeight: PartialApply[floatOptimizerFactory.Constructor, Witness.`"weight"`.T]

  @inject
  protected def floatWeightParameter: FloatWeight <:< floatPartialApplyWeight.Parameter

  @(inject @getter)
  protected val floatPartialApplyOriginalDelta: PartialApply[floatPartialApplyWeight.Rest, Witness.`"originalDelta"`.T]

  @inject
  protected def floatOriginalDeltaParameter: scala.Float <:< floatPartialApplyOriginalDelta.Parameter

  @(inject @getter)
  protected val floatWeightFactory: Factory[FloatWeight]

  @(inject @getter)
  protected val floatPartialApplyData: PartialApply[floatWeightFactory.Constructor, Witness.`"data"`.T]

  @inject
  protected def floatDataParameter: scala.Float <:< floatPartialApplyData.Parameter

}
