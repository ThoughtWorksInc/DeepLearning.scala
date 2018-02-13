package com.thoughtworks.deeplearning.plugins

import com.dongxiguo.fastring.Fastring.Implicits._
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
    : Do[DeviceBuffer[Float] => Do[Unit]] <:< floatDeviceBufferPartialApplyOriginalDelta.Parameter

  trait FloatDeviceBufferWeightApi extends DeviceBufferWeightApi {
    this: FloatDeviceBufferWeight =>
    type Element = Float

    protected type PartiallyAppliedOptimizer = floatDeviceBufferPartialApplyOriginalDelta.Rest

    /** @usecase def backward(delta: Delta): Do[Unit] = ???
      */
    protected def backward[SubtypeOfOptimizer](originalDelta: Do[DeviceBuffer[Float] => Do[Unit]])(
        implicit implicitApplyRest: ImplicitApply.Aux[floatDeviceBufferPartialApplyOriginalDelta.Rest,
                                                      SubtypeOfOptimizer],
        asOptimizer: SubtypeOfOptimizer <:<
          OptimizerApi {
            type Delta <: DeviceBuffer[Float] => Do[Unit]
          }): Do[Unit] = {
      val optimizer: OptimizerApi {
        type Delta <: DeviceBuffer[Float] => Do[Unit]
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
      allocateBuffer[Float](scalarDeltaSize * data.length).intransitiveFlatMap { delta =>
        doDelta.flatMap { setDelta =>
          setDelta(delta).flatMap { _: Unit =>
            // TODO: Add a lock for data
            subtractInplace(data, delta)
          }
        }
      }
    }
  }

  def subtractInplace(data: DeviceBuffer[Float], delta: DeviceBuffer[Float]): Do[Unit] = {
    Do.monadicCloseable(subtractInplaceProgram.createFirstKernel())
      .flatMap { kernel =>
        kernel(0) = data
        kernel(1) = delta
        kernel(2) = delta.length
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

    type Delta = DeviceBuffer[Float] => Do[Unit]

    val weight: FloatDeviceBufferWeight

  }

  /** @template */
  type FloatDeviceBufferOptimizer <: FloatDeviceBufferOptimizerApi with Optimizer

  /** For performance purpose, the delta of a scalar value are not summed to a scalar until the delta is used.
    *
    * Instead, the delta is stored as a small array of [[scalarDeltaSize]] items,
    * by summing a very large array to this small array.
    * When running a kernel that produce the small array,
    * both the global size and local size can be [[scalarDeltaSize]].
    *
    * @note `scalarDeltaSize` should be a multiple of the warp size (NVIDIA) or the wavefront size (AMD),
    *       which can be determined by invoke [[org.lwjgl.opencl.CL10.clGetKernelInfo clGetKernelInfo]]
    *       with [[org.lwjgl.opencl.CL11.CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE]].
    *
    */
  val scalarDeltaSize = 256

  private lazy val subtractInplaceProgram: Program = {
    val program = createProgramWithSource(
      fast"""
        kernel void subtract_inplace(global float * restrict data, global const float (* restrict delta)[$scalarDeltaSize]) {
          global const float (* restrict currentDelta)[$scalarDeltaSize] = delta + get_global_id(0)
          float total_delta = 0.0f;
          for (int i = 0; i < $scalarDeltaSize; i++) {
            total_delta += (*currentDelta)[i];
          }
          data[get_global_id(0)] -= total_delta;
        }
      """
    )

    program.build()
    program
  }

}
