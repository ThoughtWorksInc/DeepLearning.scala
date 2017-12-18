package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.compute.Memory
import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply}
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.feature.ImplicitApply.Aux
import com.thoughtworks.raii.asynchronous._
import com.thoughtworks.future._

import shapeless.Witness

import scalaz.syntax.all._

trait FloatDeviceBufferWeights extends DeviceBufferWeights with Weights {

  // Workaround for https://github.com/milessabin/shapeless/issues/755
  implicit private def witnessThis: Witness.Aux[this.type] = Witness.mkWitness(this)

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
    : Do[DeviceBuffer[Float]] <:< floatDeviceBufferPartialApplyOriginalDelta.Parameter

  trait FloatDeviceBufferWeightApi extends DeviceBufferWeightApi {
    this: FloatDeviceBufferWeight =>
    type Element = Float

    protected type PartiallyAppliedOptimizer = floatDeviceBufferPartialApplyOriginalDelta.Rest

    /** @usecase def backward(delta: Delta): Do[Unit] = ???
      */
    protected def backward[SubtypeOfOptimizer](originalDelta: Do[DeviceBuffer[Float]])(
        implicit implicitApplyRest: ImplicitApply.Aux[floatDeviceBufferPartialApplyOriginalDelta.Rest,
                                                      SubtypeOfOptimizer],
        asOptimizer: SubtypeOfOptimizer <:<
          OptimizerApi {
            type Delta <: DeviceBuffer[Float]
          }): Do[Unit] = {
      val optimizer: OptimizerApi {
        type Delta <: DeviceBuffer[Float]
      } =
        asOptimizer(
          implicitApplyRest(
            floatDeviceBufferPartialApplyOriginalDelta(
              floatDeviceBufferPartialApplyWeight(
                floatDeviceBufferOptimizerFactory.newInstance,
                floatDeviceBufferWeightParameter(this)
              ),
              floatDeviceBufferOriginalDeltaParameter(originalDelta)
            )))

      val doDelta = optimizer.delta
      doDelta.intransitiveFlatMap { delta =>
        // TODO: synchronize
        subtractInplace(data, delta)
      }

    }

  }

  def subtractInplace(data: DeviceBuffer[Float], delta: DeviceBuffer[Float]): Do[Unit] = {
    Do.monadicCloseable(subtractInplaceProgram.createFirstKernel())
      .flatMap { kernel =>
        kernel(0) = data
        kernel(1) = delta
        val length = data.length
        val doEvent: Do[Event] = kernel.enqueue(length)
        doEvent.flatMap { event =>
          val doWait: Do[Unit] = Do.garbageCollected(event.waitForComplete())
          doWait
        }
      }
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
    def apply[Out](data: DeviceBuffer[Float])(
        implicit implicitApplyRest: ImplicitApply.Aux[floatDeviceBufferWeightPartialApplyData.Rest, Out],
        isFloatDeviceBufferWeight: Out <:< FloatDeviceBufferWeight
    ): FloatDeviceBufferWeight = {
      isFloatDeviceBufferWeight(
        implicitApplyRest(
          floatDeviceBufferWeightPartialApplyData(floatDeviceBufferWeightFactory.newInstance,
                                                  floatDeviceBufferWeightDataParameter(data))
        ))
    }
  }

  trait FloatDeviceBufferOptimizerApi extends OptimizerApi {
    this: FloatDeviceBufferOptimizer =>

    type Delta = DeviceBuffer[Float]

    val weight: FloatDeviceBufferWeight

  }

  /** @template */
  type FloatDeviceBufferOptimizer <: FloatDeviceBufferOptimizerApi with Optimizer

  private lazy val subtractInplaceProgram: Program = {
    val program = createProgramWithSource(
      Seq("""
        kernel void subtract_inplace(global float* restrict input0, global const float* restrict input1) {
          const size_t index = get_global_id(0);
          input0[index] -= input1[index];
        }
      """)
    )

    program.build()
    program
  }

}
