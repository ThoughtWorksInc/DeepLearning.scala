package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply}
import com.thoughtworks.future._
import com.thoughtworks.raii.asynchronous._
import com.thoughtworks.tryt.covariant.TryT

import scala.util.{Failure, Success}
import scalaz.syntax.bind._
import com.thoughtworks.continuation._

/** A plugin that enables [[Weight]] in neural networks.
  *
  * @author 杨博 (Yang Bo)
  */
trait Weights extends Differentiables {

  trait WeightApi extends DifferentiableApi {

    protected type PartiallyAppliedOptimizer

    /** @usecase def forward: Do[Tape[Data, Delta] ] = ???
      */
    final def forward[SubtypeOfOptimizer](
        implicit implicitApplyRest: ImplicitApply.Aux[PartiallyAppliedOptimizer, SubtypeOfOptimizer],
        asOptimizer: SubtypeOfOptimizer <:< OptimizerApi { type Delta <: WeightApi.this.Delta })
      : Do[Tape[Data, Delta]] = {
      Do.now(
        Tape[Data, Delta](
          data, { doDelta: Do[Delta] =>
            val doUpdate: Do[Unit] = doDelta.intransitiveFlatMap(backward(_))
            val Future(TryT(continuation)) = doUpdate.run
            continuation.flatMap {
              case Success(()) => UnitContinuation.now(())
              case Failure(e)  => handleException(e)
            }
          }
        ))
    }

    /** @usecase def backward(delta: Delta): Do[Unit] = ???
      */
    protected def backward[SubtypeOfOptimizer](delta: Delta)(
        implicit implicitApplyRest: ImplicitApply.Aux[PartiallyAppliedOptimizer, SubtypeOfOptimizer],
        asOptimizer: SubtypeOfOptimizer <:< OptimizerApi { type Delta <: WeightApi.this.Delta }): Do[Unit]

    type Data
    type Delta

    var data: Data

  }

  /** @template */
  type Weight <: WeightApi with Differentiable

  trait OptimizerApi {
    type Delta

    protected val originalDelta: Delta
    def delta: Delta = originalDelta
  }

  /** @template */
  type Optimizer <: OptimizerApi

  trait ImplicitsApi {

    implicit def weightDeepLearning[SubtypeOfWeight, Data0, Delta0, OriginalDeltaRest, SubtypeOfOptimizer](
        implicit asWeight: SubtypeOfWeight <:<
          WeightApi {
            type Data = Data0
            type Delta = Delta0
            type PartiallyAppliedOptimizer = OriginalDeltaRest
          },
        implicitApplyRest: ImplicitApply.Aux[OriginalDeltaRest, SubtypeOfOptimizer],
        asOptimizer: SubtypeOfOptimizer <:< OptimizerApi { type Delta = Delta0 }
    ): DeepLearning.Aux[SubtypeOfWeight, Data0, Delta0] = {
      new DeepLearning[SubtypeOfWeight] {
        override type Data = Data0
        override type Delta = Delta0

        override def forward(subtypeOfWeight: SubtypeOfWeight): Do[Tape[Data0, Delta0]] = {
          asWeight(subtypeOfWeight).forward[SubtypeOfOptimizer]
        }
      }
    }
  }

  /** @template */
  type Implicits <: ImplicitsApi

}
