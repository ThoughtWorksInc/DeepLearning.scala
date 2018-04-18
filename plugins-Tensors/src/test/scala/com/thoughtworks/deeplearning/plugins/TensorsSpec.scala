package com.thoughtworks.deeplearning.plugins

import java.nio.ByteBuffer

import com.thoughtworks.compute.{Memory, OpenCL}
import com.thoughtworks.feature.Factory
import TensorsSpec._
import com.thoughtworks.raii.asynchronous._
import org.lwjgl.opencl.CLCapabilities

import scala.language.existentials

/**
  * @author 杨博 (Yang Bo)
  */
class TensorsSpec {
  private val hyperparameters =
    Factory[
      OpenCL.GlobalExecutionContext with OpenCL.UseAllDevices with OpenCL.UseFirstPlatform with OpenCL.CommandQueuePool with Tensors]
      .newInstance(
        handleOpenCLNotification = handleOpenCLNotification,
        numberOfCommandQueuesForDevice = { (deviceId: Long, capabilities: CLCapabilities) =>
          1
        }
      )
}

object TensorsSpec {

  private val handleOpenCLNotification = { (errorInfo: String, buffer: ByteBuffer) =>
    if (buffer.remaining > 0) {
      val hexText = for (i <- (buffer.position until buffer.limit).view) yield {
        f"${buffer.get(i)}%02X"
      }
      Console.err.println(hexText.mkString(errorInfo, " ", ""))
      Console.err.flush()
    } else {
      Console.err.println(errorInfo)
      Console.err.flush()
    }
  }
}
