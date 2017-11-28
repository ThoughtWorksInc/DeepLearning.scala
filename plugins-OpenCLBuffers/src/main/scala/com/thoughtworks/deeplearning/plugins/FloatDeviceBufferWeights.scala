package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply}
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.feature.ImplicitApply.Aux
import com.thoughtworks.raii.asynchronous.Do
import shapeless.Witness

trait FloatDeviceBufferWeights extends DeviceBufferWeights with Weights {

  trait ImplicitsApi extends super[DeviceBufferWeights].ImplicitsApi with super[Weights].ImplicitsApi

  type Implicits <: ImplicitsApi

  @inject
  protected val floatDeviceBufferOptimizerFactory: Factory[FloatDeviceBufferOptimizer]

  @inject
  protected val floatDeviceBufferPartialApplyWeight: PartialApply[floatDeviceBufferOptimizerFactory.Constructor,
                                                                  Witness.`"weight"`.T]

  @inject
  protected def floatDeviceBufferWeightParameter
    : FloatDeviceBufferWeight <:< floatDeviceBufferPartialApplyWeight.Parameter

  @inject
  protected val floatDeviceBufferPartialApplyOriginalDelta: PartialApply[floatDeviceBufferPartialApplyWeight.Rest,
                                                                         Witness.`"originalDelta"`.T]

  @inject
  protected def floatDeviceBufferOriginalDeltaParameter
    : DeviceBuffer[Float] <:< floatDeviceBufferPartialApplyOriginalDelta.Parameter

  trait FloatDeviceBufferWeightApi extends DeviceBufferWeightApi {
    this: FloatDeviceBufferWeight =>
    type Element = Float

    protected type PartiallyAppliedOptimizer = floatDeviceBufferPartialApplyOriginalDelta.Rest

    /** @usecase def backward(delta: Delta): Do[Unit] = ???
      */
    protected def backward[SubtypeOfOptimizer](originalDelta: DeviceBuffer[Float])(
        implicit implicitApplyRest: ImplicitApply.Aux[floatDeviceBufferPartialApplyOriginalDelta.Rest,
                                                      SubtypeOfOptimizer],
        asOptimizer: SubtypeOfOptimizer <:<
          OptimizerApi {
            type Delta <: DeviceBuffer[Float]
          }): Do[Unit] = {

      val optimizer: OptimizerApi {
        type Delta <: DeviceBuffer[Float]
      } = asOptimizer(
        implicitApplyRest(
          floatDeviceBufferPartialApplyOriginalDelta(
            floatDeviceBufferPartialApplyWeight(
              floatDeviceBufferOptimizerFactory.newInstance,
              floatDeviceBufferWeightParameter(this)
            ),
            floatDeviceBufferOriginalDeltaParameter(originalDelta)
          )))

      val delta = optimizer.delta

      // FIXME: should use queue
//
//      data -= delta
      ???
    }
//
//    /** @usecase def backward(delta: Delta): Do[Unit] = ???
//      */
//    override protected def backward[SubtypeOfOptimizer](originalDelta: DeviceBuffer[Float])(
//        implicit implicitApplyRest: ImplicitApply.Aux[floatDeviceBufferPartialApplyOriginalDelta.Rest, SubtypeOfOptimizer],
//        asOptimizer: SubtypeOfOptimizer <:<
//          OptimizerApi {
//            type Delta <: Do[DeviceBuffer[Float]]
//          }): Do[Unit] = {
////      val optimizer: OptimizerApi {
////        type Delta <: Do[DeviceBuffer[Float]]
////      } = asOptimizer(
////        implicitApplyRest(
////          floatDeviceBufferPartialApplyOriginalDelta(
////            floatDeviceBufferPartialApplyWeight(
////              floatDeviceBufferOptimizerFactory.newInstance,
////              floatDeviceBufferWeightParameter(this)
////            ),
////            floatDeviceBufferOriginalDeltaParameter(originalDelta)
////          )))
////
////      val delta = optimizer.delta
//
//      ???
////      val rest: partialApply.Rest = partialApply(floatDeviceBufferOptimizerFactory.newInstance, thisIsParameter(this))
////
////      val optimizer: FloatDeviceBufferOptimizer = outIsOptimizer(implicitApplyRest(rest))
////
////      ???
////      Do.delay {
////        val optimizer: FloatDeviceBufferOptimizer = ???
////        val delta = optimizer.delta
////        synchronized {
////          data -= delta
////        }
////      }
//    }
  }

  type FloatDeviceBufferWeight <: DeviceBufferWeight with FloatDeviceBufferWeightApi

  @inject
  protected val floatDeviceBufferWeightFactory: Factory[FloatDeviceBufferWeight]

  @inject
  protected val floatDeviceBufferWeightPartialApplyData: PartialApply[floatDeviceBufferWeightFactory.Constructor,
                                                                      Witness.`"data"`.T]
  @inject
  protected def floatDeviceBufferWeightDataParameter
    : DeviceBuffer[Float] <:< floatDeviceBufferWeightPartialApplyData.Parameter


  object FloatDeviceBufferWeight {
    def apply[Out <: FloatDeviceBufferWeight](data: DeviceBuffer[Float])(
        implicit implicitApplyRest: ImplicitApply.Aux[floatDeviceBufferWeightPartialApplyData.Rest, Out]
    ): FloatDeviceBufferWeight = {
      implicitApplyRest(
        floatDeviceBufferWeightPartialApplyData(floatDeviceBufferWeightFactory.newInstance,
                                                floatDeviceBufferWeightDataParameter(data))
      )
    }
  }

  trait FloatDeviceBufferOptimizerApi extends OptimizerApi {
    this: FloatDeviceBufferOptimizer =>

    type Delta = DeviceBuffer[Float]

    val weight: FloatDeviceBufferWeight

  }

  /** @template */
  type FloatDeviceBufferOptimizer <: FloatDeviceBufferOptimizerApi with Optimizer

//  @inject
//  protected val floatDeviceBufferOptimizerFactory: Factory[FloatDeviceBufferOptimizer]

  //def xxx: FloatDeviceBufferOptimizer = ???
  //
  //  trait ImplicitsApi extends super[Weights].ImplicitsApi with super[OpenCL].ImplicitsApi {
  //    implicit def deviceBufferWeightDeepLearning[Element]
  //    : DeepLearning.Aux[DeviceBufferWeight[Element], DeviceBuffer[Element], DeviceBuffer[Element]] = ???
  //  }
}
