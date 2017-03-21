package com.thoughtworks.deeplearning

import java.nio.FloatBuffer

import org.lwjgl.opencl.{CL10, CL12, CL20}
import CL10._
import CL12._
import CL20._
import com.qifun.statelessFuture.Future
import com.qifun.statelessFuture.util.Promise
import com.thoughtworks.deeplearning.DifferentiableKernel.DifferentiableExpression
import com.thoughtworks.deeplearning.Symbolic.Layers.Literal
import org.lwjgl.BufferUtils
import org.scalatest.{AsyncFreeSpec, FreeSpec, Matchers}
import shapeless.HNil

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
class DifferentiableKernelSpec extends AsyncFreeSpec with Matchers {

  private def toConcurrentFuture[A](statelessFuture: Future.Stateless[A]): scala.concurrent.Future[A] = {
    val p = Promise[A]
    p.completeWith(statelessFuture)
    p
  }

  "*" in toConcurrentFuture {
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
    val dk = DifferentiableExpression.FloatLiteral(42.0f).compile(context, device, commandQueue, executionContext)
    Future {
      // TODO: Literal
      val outputTape = dk.forward((1, Map.empty)).await
      try {
        val f = BufferUtils.createFloatBuffer(1)
        val r = commandQueue.readBuffer(outputTape.value, f)
        r.await
        f.capacity should be(1)
        f.get(0) should be(42.0f)
      } finally {
        outputTape.close()
      }
    }
  }

}
