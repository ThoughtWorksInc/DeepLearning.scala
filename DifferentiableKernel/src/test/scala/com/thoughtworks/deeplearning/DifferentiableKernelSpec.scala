package com.thoughtworks.deeplearning

import org.lwjgl.opencl.{CL10, CL12, CL20}
import CL10._
import CL12._
import CL20._
import com.thoughtworks.deeplearning.DifferentiableKernel.OpenCLLayer
import com.thoughtworks.raii.future.Do
import com.thoughtworks.raii.future.Do._
import scalaz.syntax.all._
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
    val openclTask = throwableMonadic[Do] {
      val supportedProperties = device.queueProperties

      val context = Do
        .managed(platform.createContext({ (errorInfo, data) =>
          info(errorInfo)
        }, device))
        .each

      val commandQueue = Do
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
    val semaphore = AsynchronousSemaphore(3)

    "When filling a buffer with 42.0f" - {
      val differentiableKernel = {
        import OpenCLLayer._
        floatLiteral(42.0f)
      }
      import DifferentiableKernel._
      import StaticDslType._
      val resultTask = Do
        .run(
          openclTask
            .flatMap {
              case (context, commandQueue) =>
                Do.unmanaged(
                  Do.run(
                    throwableMonadic[Do] {
                      val layer = differentiableKernel.compile(context, commandQueue, semaphore).each
                      val outputTape = layer(1, HNil).each
                      val delta = Do.managed(context.createBuffer[Float](1)).map(PendingBuffer(_, Nil))
                      Do.unmanaged(outputTape.backward(delta)).each
                      val f = BufferUtils.createFloatBuffer(1)
                      val event = Do
                        .managed(commandQueue.enqueueReadBuffer(outputTape.data.buffer, f, outputTape.data.events: _*))
                        .each
                      Do.unmanaged(event.waitForComplete()).each
                      f
                    }
                  )
                )
            })
        .get
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

    "When filling a buffer with another buffer" in {

      val differentiableKernel = {
        import OpenCLLayer._
        import DifferentiableKernel._
        import StaticDslType._
        getElement(bufferIdentifier[Float, Float]('input), getGlobalId(intLiteral(0)))
      }

      import DifferentiableKernel._
      import StaticDslType._

      val resultTask =
        Do.run(openclTask.flatMap {
            case (context, commandQueue) =>
              Do.unmanaged(
                Do.run(
                  throwableMonadic[Do] {
                    val layer = differentiableKernel.compile(context, commandQueue, semaphore).each
                    val outputTape = layer(1, ??? :: HNil).each
                    val delta = Do.managed(context.createBuffer[Float](1)).map(PendingBuffer(_, Nil))
                    Do.unmanaged(outputTape.backward(delta)).each
                    val f = BufferUtils.createFloatBuffer(1)
                    val event = Do
                      .managed(commandQueue.enqueueReadBuffer(outputTape.data.buffer, f, outputTape.data.events: _*))
                      .each
                    Do.unmanaged(event.waitForComplete()).each
                    f
                  }
                )
              )
          })
          .get

      true should be(true)
//      differentiableKernel.compile(???, device, ???)

    }

  }
}
