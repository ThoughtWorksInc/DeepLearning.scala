package com.thoughtworks.deeplearning

import org.lwjgl.opencl.{CL10, CL12, CL20}
import CL10._
import CL12._
import CL20._
import com.thoughtworks.deeplearning.DifferentiableKernel.DifferentiableExpression
import com.thoughtworks.raii.RAIITask
import org.lwjgl.BufferUtils
import org.scalatest.{Assertion, AsyncFreeSpec, Matchers}
import com.thoughtworks.each.Monadic._

import scala.concurrent.Promise
import scalaz.concurrent.Future

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
class DifferentiableKernelSpec extends AsyncFreeSpec with Matchers {
  executionContext
  "*" in {
    val platform = OpenCL.platforms.head

    val device = platform.devices.maxBy { device =>
      Seq(CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR).indexOf(device.deviceType)
    }

    val Array(supportedProperties) = {
      val propertiesBuffer = Array[Long](0L)
      clGetDeviceInfo(device.id, CL_DEVICE_QUEUE_PROPERTIES, propertiesBuffer, null) match {
        case CL_SUCCESS =>
          propertiesBuffer
        case errorCode =>
          throw new IllegalStateException(s"clGetDeviceInfo error: $errorCode")
      }
    }
    val context = { () =>
      platform.createContext({ (errorInfo, data) =>
        println(errorInfo)
      }, device)
    }.apply()
    val commandQueue =
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
      )
    val layerTask: RAIITask[(Int, Map[Any, Tape]) => RAIITask[Tape.Aux[OpenCL.Buffer[Float], OpenCL.Buffer[Float]]]] =
      DifferentiableExpression.FloatLiteral(42.0f).compile(context, device, commandQueue)

    val p = Promise[Assertion]
    throwableMonadic[RAIITask] {
      val layer = layerTask.each
      val outputTape = layer(1, Map.empty).each
      val delta = RAIITask.managed(context.createBuffer[Float](1))
      RAIITask.unmanaged(outputTape.backward(delta)).each
      val f = BufferUtils.createFloatBuffer(1)
      val event = RAIITask.managed(commandQueue.enqueueReadBuffer(outputTape.data, f)).each
      RAIITask.unmanaged(event.waitForComplete()).each
      f.capacity should be(1)
      f.get(0) should be(42.0f)
    }.run.run.unsafePerformAsync { either =>
      p.complete(scalaz.std.`try`.fromDisjunction(either))
    }

    p.future
  }

}
