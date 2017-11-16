package com.thoughtworks.deeplearning.plugins

import java.nio.ByteBuffer

import com.thoughtworks.compute.OpenCL
import com.thoughtworks.feature.Factory
import com.thoughtworks.feature.mixins.ImplicitsSingleton
import org.lwjgl.opencl.CLCapabilities
import com.thoughtworks.continuation._
import com.thoughtworks.raii.asynchronous._
import com.thoughtworks.each.Monadic._
import scalaz.syntax.all._

/**
  * @author 杨博 (Yang Bo)
  */
class OpenCLBufferLiteralsSpec {

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

  def test: Do[Unit] = {
    /*
    val doHyperparameters = Do.monadicCloseable(Factory[
      OpenCLBufferLiterals with Training with ImplicitsSingleton with OpenCL.UseAllDevices with OpenCL.UseFirstPlatform with OpenCL.CommandQueuePool]
      .newInstance(
        handleOpenCLNotification = handleOpenCLNotification,
        numberOfCommandQueuesForDevice = { (deviceId: Long, capabilities: CLCapabilities) =>
          1
        }
      )
    import hyperparameters.implicits._
    val myFloatBuffer = hyperparameters.allocateBuffer[Float](1024)
    predictValue = myFloatBuffer.predict
    println(predictValue)
     */



    val doHyperparameters = Do.monadicCloseable(Factory[
      OpenCLBufferLiterals with Training with ImplicitsSingleton with OpenCL.UseAllDevices with OpenCL.UseFirstPlatform with OpenCL.CommandQueuePool]
      .newInstance(
        handleOpenCLNotification = handleOpenCLNotification,
        numberOfCommandQueuesForDevice = { (deviceId: Long, capabilities: CLCapabilities) =>
          1
        }
      ))

    doHyperparameters.flatMap { hyperparameters =>
      import hyperparameters.implicits._
      hyperparameters.allocateBuffer[Float](1024).flatMap { myFloatBuffer =>
        Do.garbageCollected(myFloatBuffer.predict).map { predictValue =>
          println(predictValue)
        }
      }
    }

  }

}
