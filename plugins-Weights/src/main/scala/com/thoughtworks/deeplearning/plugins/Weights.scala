package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply}
import com.thoughtworks.raii.asynchronous.Do
import shapeless.Witness

import scalaz.{-\/, \/-}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait Weights {

  trait WeightApi { self =>
    protected type Optimizer <: Weights.this.Optimizer {
      type Weight >: self.type
      type Delta = self.Delta
    }

    type Data
    type Delta
    private[Weights] def handleExceptionFriend(throwable: Throwable): Unit = handleException(throwable)
    protected def handleException(throwable: Throwable): Unit = ()
    var data: Data

  }

  type Weight <: WeightApi

  object Weight {
    type Aux[Optimizer0, Data0, Delta0] = Weight {
      type Data = Data0
      type Delta = Delta0
      type Optimizer = Optimizer0
    }
  }

  trait OptimizerApi { self =>
    type Delta
    protected type Weight <: Weights.this.Weight {
      type Optimizer >: self.type
    }

    protected val weight: Weight

    protected val originalDelta: Delta
    protected def delta: Delta = originalDelta
    protected def update(): Do[Unit]
    private[Weights] def updateFriend(): Do[Unit] = update()
  }

  type Optimizer <: OptimizerApi

  trait ImplicitsApi {
    implicit def weightDeepLearning[SubtypeOfWeight,
                                         Optimizer0 <: Optimizer,
                                         Data0,
                                         Delta0,
                                         OptimizerConstructor,
                                         WeightParameter,
                                         WeightRest,
                                         OriginalDeltaParameter,
                                         OriginalDeltaRest,
                                         SubtypeOfOptimizer](
        implicit asWeight: SubtypeOfWeight <:< Weight.Aux[Optimizer0, Data0, Delta0],
        factory: Factory.Aux[Optimizer0, OptimizerConstructor],
        partialApplyWeight: PartialApply.Aux[OptimizerConstructor, Witness.`"weight"`.T, WeightParameter, WeightRest],
        asWeightParameter: SubtypeOfWeight <:< WeightParameter,
        partialApplyOriginalDelta: PartialApply.Aux[WeightRest,
                                                    Witness.`"originalDelta"`.T,
                                                    OriginalDeltaParameter,
                                                    OriginalDeltaRest],
        asOriginalDeltaParameter: Delta0 <:< OriginalDeltaParameter,
        implicitApplyRest: ImplicitApply.Aux[OriginalDeltaRest, SubtypeOfOptimizer],
        asOptimizer: SubtypeOfOptimizer <:< Optimizer0
    ): DeepLearning.Aux[SubtypeOfWeight, Data0, Delta0] = {
      new DeepLearning[SubtypeOfWeight] {
        override type Data = Data0
        override type Delta = Delta0

        override def forward(subtypeOfWeight: SubtypeOfWeight): Do[Tape[Data0, Delta0]] = {
          val weight = asWeight(subtypeOfWeight)
          Do.now(
            Tape[Data0, Delta0](
              weight.data, { doDelta: Do[Delta0] =>
                val doUpdate: Do[Unit] = Do.releaseFlatMap(doDelta) { delta =>
                  asOptimizer(
                    implicitApplyRest(
                      partialApplyOriginalDelta(partialApplyWeight(factory.newInstance,
                                                                   asWeightParameter(subtypeOfWeight)),
                                                asOriginalDeltaParameter(delta)))).updateFriend()

                }
                Do.run(doUpdate).get.map {
                  case \/-(()) => ()
                  case -\/(e) => weight.handleExceptionFriend(e)
                }
              }
            ))
        }
      }
    }
  }

  type Implicits <: ImplicitsApi

}
