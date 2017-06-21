package com.thoughtworks.deeplearning
package plugins
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply}
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import shapeless.Witness

import annotation.meta.getter
import scalaz.{-\/, \/-}
import scalaz.concurrent.Future

/**
  * @author 杨博 (Yang Bo)
  */
trait FloatWeights extends Weights {

  trait FloatWeightApi extends WeightApi { this: FloatWeight =>

    override type Delta = Float
    override type Data = Float
    override protected type Optimizer = FloatOptimizer

  }
  type FloatWeight <: FloatWeightApi with Weight

  @inject
  protected val floatWeightFactory: Factory[FloatWeight]

  @inject
  protected val floatPartialApplyData: PartialApply[floatWeightFactory.Constructor, Witness.`"data"`.T]

  @inject
  protected def floatDataParameter: Float <:< floatPartialApplyData.Parameter
  object FloatWeight extends {
    def apply[SubtypeOfWeight, OptimizerFunction, Optimizer](data: Float)(
        implicit implicitApplyRest: ImplicitApply[floatPartialApplyData.Rest]) = {
      implicitApplyRest(floatPartialApplyData(floatWeightFactory.newInstance, floatDataParameter(data)))
    }
  }

  trait FloatOptimizerApi extends OptimizerApi { this: FloatOptimizer =>

    type Delta = Float

    override protected type Weight = FloatWeight

    override protected def update() = {
      Do.delay {
        weight.synchronized {
          weight.data -= delta
        }
      }
    }

  }
  type FloatOptimizer <: FloatOptimizerApi with Optimizer

}
