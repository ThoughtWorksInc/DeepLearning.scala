package com.thoughtworks.deeplearning
package plugins
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply}
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import org.nd4j.linalg.api.ndarray.INDArray
import shapeless.Witness
import org.nd4s.Implicits._
import com.thoughtworks.feature.mixins.ImplicitsSingleton

import annotation.meta.getter
import scala.concurrent.ExecutionContext
import scalaz.syntax.all._

/** A plugin to create [[org.nd4j.linalg.api.ndarray.INDArray]] weights.
  *
  * @note Custom optimization algorithm for updating [[INDArrayWeight]] can be implemented by creating a plugin
  *       that provides a overridden [[INDArrayOptimizer]] that provides an overridden [[INDArrayOptimizer.delta]].
  *
  * @author 杨博 (Yang Bo)
  */
trait INDArrayWeights extends Weights with ImplicitsSingleton {

  @inject
  implicit protected def deepLearningExecutionContext: ExecutionContext

  override type Implicits <: ImplicitsApi

  import implicits._

  trait INDArrayWeightApi extends WeightApi { this: INDArrayWeight =>

    override type Delta = INDArray
    override type Data = INDArray

    override protected type PartiallyAppliedOptimizer = indArrayPartialApplyOriginalDelta.Rest

    override protected def backward[SubtypeOfOptimizer](originalDelta: INDArray)(
        implicit implicitApplyRest: ImplicitApply.Aux[PartiallyAppliedOptimizer, SubtypeOfOptimizer],
        asOptimizer: SubtypeOfOptimizer <:< OptimizerApi { type Delta <: INDArray }): Do[Unit] = {

      Do.execute {
        val delta =
          implicitApplyRest(
            indArrayPartialApplyOriginalDelta(indArrayPartialApplyWeight(indArrayOptimizerFactory.newInstance,
                                                                         indArrayWeightParameter(this)),
                                              indArrayOriginalDeltaParameter(originalDelta))).delta

        synchronized {
          data -= delta
          ()
        }
      }

    }

  }

  /** @template */
  type INDArrayWeight <: INDArrayWeightApi with Weight

  @inject
  protected val indArrayWeightFactory: Factory[INDArrayWeight]

  @inject
  protected val indArrayPartialApplyData: PartialApply[indArrayWeightFactory.Constructor, Witness.`"data"`.T]

  @inject
  protected def indArrayDataParameter: INDArray <:< indArrayPartialApplyData.Parameter
  object INDArrayWeight {

    /** @usecase def apply(data: Float): INDArrayWeight = ???
      */
    def apply[SubtypeOfWeight, OptimizerFunction, Optimizer](data: INDArray)(
        implicit implicitApplyRest: ImplicitApply[indArrayPartialApplyData.Rest]) = {
      implicitApplyRest(indArrayPartialApplyData(indArrayWeightFactory.newInstance, indArrayDataParameter(data)))
    }
  }

  trait INDArrayOptimizerApi extends OptimizerApi { this: INDArrayOptimizer =>

    override type Delta = INDArray

    val weight: INDArrayWeight

  }

  /** @template */
  type INDArrayOptimizer <: Optimizer with INDArrayOptimizerApi

  @inject
  protected val indArrayOptimizerFactory: Factory[INDArrayOptimizer]
  @inject
  protected val indArrayPartialApplyWeight: PartialApply[indArrayOptimizerFactory.Constructor, Witness.`"weight"`.T]
  @inject
  protected def indArrayWeightParameter: INDArrayWeight <:< indArrayPartialApplyWeight.Parameter

  @inject
  protected val indArrayPartialApplyOriginalDelta: PartialApply[indArrayPartialApplyWeight.Rest,
                                                                Witness.`"originalDelta"`.T]

  @inject
  protected def indArrayOriginalDeltaParameter: INDArray <:< indArrayPartialApplyOriginalDelta.Parameter

}
