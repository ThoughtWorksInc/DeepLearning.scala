package com.thoughtworks.deeplearning
package plugins
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply}
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import org.nd4j.linalg.api.ndarray.INDArray
import shapeless.Witness
import org.nd4s.Implicits._

import annotation.meta.getter
import scala.concurrent.ExecutionContext
import scalaz.syntax.all._

/**
  * @author 杨博 (Yang Bo)
  */
trait INDArrayWeights extends Weights with ImplicitsSingleton {

  @inject
  implicit protected def deepLearningExecutionContext: ExecutionContext

  type Implicits <: ImplicitsApi

  import implicits._

  trait INDArrayWeightApi extends WeightApi { this: INDArrayWeight =>

    type Delta = INDArray
    override type Data = INDArray
    override protected type Optimizer = INDArrayOptimizer

  }
  type INDArrayWeight <: INDArrayWeightApi with Weight

  @inject
  protected val indArrayWeightFactory: Factory[INDArrayWeight]

  @inject
  protected val indArrayPartialApplyData: PartialApply[indArrayWeightFactory.Constructor, Witness.`"data"`.T]

  @inject
  protected def indArrayDataParameter: INDArray <:< indArrayPartialApplyData.Parameter
  object INDArrayWeight extends {
    def apply[SubtypeOfWeight, OptimizerFunction, Optimizer](data: INDArray)(
        implicit implicitApplyRest: ImplicitApply[indArrayPartialApplyData.Rest]) = {
      implicitApplyRest(indArrayPartialApplyData(indArrayWeightFactory.newInstance, indArrayDataParameter(data)))
    }
  }

  trait INDArrayOptimizerApi extends OptimizerApi { this: INDArrayOptimizer =>

    type Delta = INDArray

    override protected type Weight = INDArrayWeight

    override protected def update(): Do[Unit] = {
      Do.jump().map { _: Unit =>
        weight.synchronized {
          weight.data -= delta
          ()
        }
      }
    }

  }
  type INDArrayOptimizer <: INDArrayOptimizerApi with Optimizer

}
