package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply}
import com.thoughtworks.raii.asynchronous.Do
import shapeless.Witness

import scalaz.{-\/, \/-}

/** A plugin that enables [[Weight]] in neural networks.
  *
  * @author 杨博 (Yang Bo)
  */
trait Weights {

  trait WeightApi {

    protected type PartiallyAppliedOptimizer

    /** @usecase def forward: Do[Tape[Data, Delta] ] = ???
      */
    final def forward[SubtypeOfOptimizer](
        implicit implicitApplyRest: ImplicitApply.Aux[PartiallyAppliedOptimizer, SubtypeOfOptimizer],
        asOptimizer: SubtypeOfOptimizer <:< Optimizer.Aux[Delta]): Do[Tape[Data, Delta]] = {
      Do.now(
        Tape[Data, Delta](
          data, { doDelta: Do[Delta] =>
            val doUpdate: Do[Unit] = Do.releaseFlatMap(doDelta) { delta =>
              backward(delta)
            }
            Do.run(doUpdate).get.map {
              case \/-(()) => ()
              case -\/(e)  => handleException(e)
            }
          }
        ))
    }

    /** @usecase def backward(delta: Delta): Do[Unit] = ???
      */
    protected def backward[SubtypeOfOptimizer](delta: Delta)(
        implicit implicitApplyRest: ImplicitApply.Aux[PartiallyAppliedOptimizer, SubtypeOfOptimizer],
        asOptimizer: SubtypeOfOptimizer <:< Optimizer.Aux[Delta]): Do[Unit]

    type Data
    type Delta

    protected def handleException(throwable: Throwable): Unit = ()

    var data: Data

  }

  /** @template */
  type Weight <: WeightApi

  object Weight {
    type Aux[PartiallyAppliedOptimizer0, Data0, Delta0] = Weight {
      type Data = Data0
      type Delta = Delta0
      type PartiallyAppliedOptimizer = PartiallyAppliedOptimizer0
    }
  }

  trait OptimizerApi {
    type Delta

    protected val originalDelta: Delta
    def delta: Delta = originalDelta
  }

  /** @template */
  type Optimizer <: OptimizerApi

  object Optimizer {
    type Aux[Delta0] = Optimizer {
      type Delta = Delta0
    }
  }

  trait ImplicitsApi {

    implicit def weightDeepLearning[SubtypeOfWeight, Data0, Delta0, OriginalDeltaRest, SubtypeOfOptimizer](
        implicit asWeight: SubtypeOfWeight <:< Weight.Aux[OriginalDeltaRest, Data0, Delta0],
        implicitApplyRest: ImplicitApply.Aux[OriginalDeltaRest, SubtypeOfOptimizer],
        asOptimizer: SubtypeOfOptimizer <:< Optimizer.Aux[Delta0]
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
