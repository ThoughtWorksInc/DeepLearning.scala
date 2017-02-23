package com.thoughtworks.deeplearning

import java.nio.ByteBuffer
import OpenCL._
import com.qifun.statelessFuture.Future
import com.thoughtworks.deeplearning.DifferentiableOpenCLBuffer.KernelLayers.Literal
import org.lwjgl.opencl.CL10._
import org.lwjgl.opencl.{CL10, CL20}
import org.scalatest._
import shapeless.HNil
import org.lwjgl.system.MemoryUtil._
import org.lwjgl.system.MemoryStack._

import com.qifun.statelessFuture.util.Promise

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DifferentiableOpenCLBufferSpec extends AsyncFreeSpec with Matchers {

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
    println(supportedProperties)
    val clContext = { () =>
      platform.createContext({ (errorInfo, data) =>
        println(errorInfo)
      }, device)
    }.apply()

    System.gc()

    System.gc()

    System.gc()
    val commandQueue =
      clContext.createCommandQueue(
        device,
        (CL10.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE & supportedProperties) |
          (if (device.capabilities.OpenCL20) {
             CL20.CL_QUEUE_ON_DEVICE
           } else { 0 })
      )

    val fill = DifferentiableOpenCLBuffer.Layers.Fill[Any, Nothing, Float, Any, HNil, HNil](clContext,
                                                                                            commandQueue,
                                                                                            Literal(1),
                                                                                            Literal(3.14f))

    Future {
      val outputBatch = fill.forward(Literal(new AnyRef)).await
      val r = commandQueue.readBuffer(outputBatch.value)
      r.await
      r.await
      val b = r.await
      println("read end")
      val f = b.asFloatBuffer
      f.capacity should be(1)
      f.get(0) should be(3.14f)
    }
  }

}
