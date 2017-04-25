package com.thoughtworks.deeplearning

import org.lwjgl.opencl.{CL10, CL12, CL20}
import CL10._
import CL12._
import CL20._
import com.thoughtworks.deeplearning.DifferentiableKernel.OpenCLLayer
import com.thoughtworks.raii.RAIITask
import org.lwjgl.BufferUtils
import org.scalatest.{Assertion, AsyncFreeSpec, Matchers}
import com.thoughtworks.each.Monadic._
import shapeless._

import scala.concurrent.Promise

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
class DifferentiableKernelSpec extends AsyncFreeSpec with Matchers {
//  def compileNil(OpenCL.Context): Unit = {}

  "Given an new OpenCL context" - {
    val platform = OpenCL.platforms.head

    val device = platform.devices.maxBy { device =>
      Seq(CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR).indexOf(device.deviceType)
    }
    val openclTask = throwableMonadic[RAIITask] {
      val supportedProperties = device.queueProperties

      val context = RAIITask
        .managed(platform.createContext({ (errorInfo, data) =>
          info(errorInfo)
        }, device))
        .each

      val commandQueue = RAIITask
        .managed(
          context.createCommandQueue(
            device,
            Map(
              CL_QUEUE_PROPERTIES -> {
                {
                  CL10.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE & supportedProperties
                } | {
                  if (device.capabilities.OpenCL20) {
                    CL_QUEUE_ON_DEVICE
                  } else {
                    0
                  }
                }
              }
            )
          ))
        .each

      (context, commandQueue)
    }

    "When fill a buffer with 42.0f" - {
      val differentiableKernel = {
        import OpenCLLayer._
        floatLiteral(42.0f)
      }
      import DifferentiableKernel._
      import StaticDslType._

      val resultTask = openclTask
        .flatMap {
          case (context, commandQueue) =>
            RAIITask.unmanaged(
              RAIITask.run(
                throwableMonadic[RAIITask] {
                  val layer = differentiableKernel.compile(context, device, commandQueue).each
                  val outputTape = layer(1, HNil).each
                  val delta = RAIITask.managed(context.createBuffer[Float](1))
                  RAIITask.unmanaged(outputTape.backward(delta)).each
                  val f = BufferUtils.createFloatBuffer(1)
                  val event = RAIITask.managed(commandQueue.enqueueReadBuffer(outputTape.data, f)).each
                  RAIITask.unmanaged(event.waitForComplete()).each
                  f
                }
              )
            )
        }
        .run
        .run
      "The content should be 42.0f" in {
        val p = Promise[Assertion]
        resultTask
          .map(_.map { f =>
            f.capacity should be(1)
            f.get(0) should be(42.0f)

          })
          .unsafePerformAsync { either =>
            p.complete(scalaz.std.`try`.fromDisjunction(either))
          }
        p.future
      }

    }

    "When fill a buffer with another buffer" in {

      val differentiableKernel = {
        import OpenCLLayer._
        import DifferentiableKernel._
        import StaticDslType._
        getElement(bufferIdentifier[Float, Float]('input), getGlobalId(intLiteral(0)))
      }

      import DifferentiableKernel._
      import StaticDslType._

      val resultTask = openclTask
        .flatMap {
          case (context, commandQueue) =>
            RAIITask.unmanaged(
              RAIITask.run(
                throwableMonadic[RAIITask] {
                  val layer = differentiableKernel.compile(context, device, commandQueue).each
                  val outputTape = layer(1, ??? :: HNil).each
                  val delta = RAIITask.managed(context.createBuffer[Float](1))
                  RAIITask.unmanaged(outputTape.backward(delta)).each
                  val f = BufferUtils.createFloatBuffer(1)
                  val event = RAIITask.managed(commandQueue.enqueueReadBuffer(outputTape.data, f)).each
                  RAIITask.unmanaged(event.waitForComplete()).each
                  f
                }
              )
            )
        }
        .run
        .run

      true should be(true)
//      differentiableKernel.compile(???, device, ???)

    }

  }
}
