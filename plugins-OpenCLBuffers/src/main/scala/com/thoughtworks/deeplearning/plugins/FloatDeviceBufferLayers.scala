package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.compute.Memory
import com.thoughtworks.continuation.UnitContinuation
import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.deeplearning.plugins.Layers.ToLayer
import com.thoughtworks.deeplearning.plugins._
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply}
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.raii.asynchronous._
import DeepLearning.ops._
import shapeless.Witness

import scalaz.syntax.all._

trait FloatDeviceBufferLayers extends DeviceBufferLayers with FloatLayers {

  // Workaround for https://github.com/milessabin/shapeless/issues/755
  implicit private def witnessThis: Witness.Aux[this.type] = Witness.mkWitness(this)

  trait FloatDeviceBufferLayerApi extends DeviceBufferLayerApi {
    type Element = Float
  }

  type FloatDeviceBufferLayer <: DeviceBufferLayer with FloatDeviceBufferLayerApi

  @inject
  protected val floatDeviceBufferLayerFactory: Factory[FloatDeviceBufferLayer]

  @inject
  protected val floatDeviceBufferLayerPartialApplyRawForward: PartialApply[floatDeviceBufferLayerFactory.Constructor,
                                                                           shapeless.Witness.`"rawForward"`.T]

  @inject
  protected def floatDeviceBufferLayerRawForwardParameter
    : Do[Tape[DeviceBuffer[Float], DeviceBuffer[Float]]] <:< floatDeviceBufferLayerPartialApplyRawForward.Parameter

  trait ImplicitsApi extends super[DeviceBufferLayers].ImplicitsApi with super[FloatLayers].ImplicitsApi {

    implicit def toFloatDeviceBufferLayer[Out <: FloatDeviceBufferLayer](
        implicit implicitApply: ImplicitApply.Aux[floatDeviceBufferLayerPartialApplyRawForward.Rest, Out])
      : Layers.ToLayer.Aux[DeviceBuffer[Float], DeviceBuffer[Float], FloatDeviceBufferLayer] =
      new Layers.ToLayer[DeviceBuffer[Float], DeviceBuffer[Float]] {
        type OutputLayer = FloatDeviceBufferLayer
        def toLayer(forward: Do[Tape[DeviceBuffer[Float], DeviceBuffer[Float]]]): FloatDeviceBufferLayer = {

          implicitApply(
            floatDeviceBufferLayerPartialApplyRawForward(floatDeviceBufferLayerFactory.newInstance,
                                                         floatDeviceBufferLayerRawForwardParameter(forward)))
        }
      }
  }

  override type Implicits <: ImplicitsApi

  def mean[Operand0, Buffer, OutputLayer](operand0: Operand0)(
      implicit
      deepLearning0: DeepLearning.Aux[Operand0, Buffer, Buffer],
      isDoBuffer: Do[Tape[Buffer, Buffer]] <:< Do[Tape[DeviceBuffer[Float], DeviceBuffer[Float]]],
      layerFactory: ToLayer.Aux[Float, Float, OutputLayer],
      memory: Memory[Float]): OutputLayer = {
    val operand0Forward: Do[Tape[DeviceBuffer[Float], DeviceBuffer[Float]]] = isDoBuffer(operand0.forward)

    val forward: Do[Tape[Float, Float]] = operand0Forward.flatMap {
      case (Tape(data0, backward0)) =>
        def outputData(data: DeviceBuffer[Float]): Do[Float] = {
          data.toHostBuffer.map { databuffer =>
            val elements: Array[Float] = memory.toArray(databuffer)
            elements.sum / elements.length
          }
        }
        def backward(doOutputDelta: Do[Float]): UnitContinuation[Unit] = {
          val delta0: Do[DeviceBuffer[Float]] = doOutputDelta.flatMap { outputDelta =>
            val length = data0.length
            allocateBuffer[Float](length).flatMap { output =>
              Do.monadicCloseable(fillValue.createFirstKernel())
                .flatMap { kernel =>
                  kernel(0) = output
                  kernel(1) = outputDelta / length
                  val self: this.type = this
                  val doEvent: Do[Event] = kernel.enqueue(length)(Witness(self))
                  doEvent.flatMap { event =>
                    val doWait: Do[Unit] = Do.garbageCollected(event.waitForComplete())
                    doWait
                  }
                }
                .intransitiveMap { _: Unit =>
                  output
                }
            }
          }

          backward0(delta0)
        }
        outputData(data0).map(Tape(_, backward))
    }

    layerFactory.toLayer(forward)
  }

  private lazy val fillValue: Program = {
    val program = createProgramWithSource(
      Seq("""
        kernel void fill_value(global float* restrict output, float value) {
          const size_t i = get_global_id(0);
          output[i] = value;
        }
      """)
    )

    program.build()
    program
  }
}
