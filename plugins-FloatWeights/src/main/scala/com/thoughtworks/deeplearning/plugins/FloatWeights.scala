package com.thoughtworks.deeplearning
package plugins
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply}
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import shapeless.Witness


/** A plugin to create [[scala.Float]] weights.
  *
  * @note Custom optimization algorithm for updating [[FloatWeight]] can be implemented by creating a plugin
  *       that provides an overridden [[FloatOptimizer]] that provides an overridden [[FloatOptimizer.delta]].
  *
  * @author 杨博 (Yang Bo)
  */
trait FloatWeights extends Weights {

  trait FloatWeightApi extends WeightApi { this: FloatWeight =>

    override type Delta = Float
    override type Data = Float

    override protected type PartiallyAppliedOptimizer = floatPartialApplyOriginalDelta.Rest

    override protected def backward[SubtypeOfOptimizer](originalDelta: Float)(
        implicit implicitApplyRest: ImplicitApply.Aux[PartiallyAppliedOptimizer, SubtypeOfOptimizer],
        asOptimizer: SubtypeOfOptimizer <:< Optimizer.Aux[Delta]): Do[Unit] = {
      Do.delay {
        val delta =
          implicitApplyRest(
            floatPartialApplyOriginalDelta(floatPartialApplyWeight(floatOptimizerFactory.newInstance,
                                                                   floatWeightParameter(this)),
                                           floatOriginalDeltaParameter(originalDelta))).delta
        synchronized {
          data -= delta
        }
      }

    }
  }

  /** @template */
  type FloatWeight <: FloatWeightApi with Weight

  @inject
  protected val floatWeightFactory: Factory[FloatWeight]

  @inject
  protected val floatPartialApplyData: PartialApply[floatWeightFactory.Constructor, Witness.`"data"`.T]

  @inject
  protected def floatDataParameter: Float <:< floatPartialApplyData.Parameter
  object FloatWeight {

    /** @usecase def apply(data: Float): FloatWeight = ???
      */
    def apply[SubtypeOfWeight, OptimizerFunction, Optimizer](data: Float)(
        implicit implicitApplyRest: ImplicitApply[floatPartialApplyData.Rest]) = {
      implicitApplyRest(floatPartialApplyData(floatWeightFactory.newInstance, floatDataParameter(data)))
    }
  }

  trait FloatOptimizerApi extends OptimizerApi { this: FloatOptimizer =>

    override type Delta = Float

    val weight: FloatWeight

  }

  /** @template */
  type FloatOptimizer <: FloatOptimizerApi with Optimizer

  @inject
  protected val floatOptimizerFactory: Factory[FloatOptimizer]

  @inject
  protected val floatPartialApplyWeight: PartialApply[floatOptimizerFactory.Constructor, Witness.`"weight"`.T]

  @inject
  protected def floatWeightParameter: FloatWeight <:< floatPartialApplyWeight.Parameter

  @inject
  protected val floatPartialApplyOriginalDelta: PartialApply[floatPartialApplyWeight.Rest, Witness.`"originalDelta"`.T]

  @inject
  protected def floatOriginalDeltaParameter: Float <:< floatPartialApplyOriginalDelta.Parameter

}
