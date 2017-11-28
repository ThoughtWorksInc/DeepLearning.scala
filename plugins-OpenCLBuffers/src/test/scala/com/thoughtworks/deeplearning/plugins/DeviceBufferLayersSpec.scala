package com.thoughtworks.deeplearning.plugins

import java.nio.{ByteBuffer, FloatBuffer}

import com.thoughtworks.compute.{Memory, OpenCL}
import com.thoughtworks.deeplearning.plugins.FloatWeightSpec.NormalDistributionRandom
import com.thoughtworks.future._
import com.thoughtworks.feature.Factory
import com.thoughtworks.feature.mixins.ImplicitsSingleton
import com.thoughtworks.raii.asynchronous.Do
import org.lwjgl.opencl.CLCapabilities
import com.thoughtworks.deeplearning.DeepLearning.Tape

import scalaz.syntax.all._
import com.thoughtworks.raii.asynchronous._
import org.scalatest._

object DeviceBufferLayersSpec {

  private[DeviceBufferLayersSpec] trait DeviceBufferOf extends OpenCL {
    def deviceBufferOf[Element](elements: Element*)(implicit memory: Memory[Element]): Do[DeviceBuffer[Element]] = {
      val hostBuffer: memory.HostBuffer = memory.allocate(elements.length)
      // TODO: optimize the performance
      for ((element, i) <- elements.view.zipWithIndex) {
        memory.put(hostBuffer, i, element)
      }
      allocateBufferFrom[Element, memory.HostBuffer](hostBuffer)(memory)
    }
  }

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

final class DeviceBufferLayersSpec extends AsyncFreeSpec /* AsyncFeatureSpec with GivenWhenThen */ with Matchers {
  import DeviceBufferLayersSpec._

  private def configure =
    Do.monadicCloseable(Factory[
      DeviceBufferOf with FloatLayers with OpenCLBufferLiterals with FloatTraining with ImplicitsSingleton with OpenCL.UseAllDevices with OpenCL.UseFirstPlatform with OpenCL.CommandQueuePool with DeviceBufferWeights with FloatDeviceBufferWeights with DeviceBufferLayers with FloatDeviceBufferLayers with NormalDistributionRandom]
      .newInstance(
        handleOpenCLNotification = handleOpenCLNotification,
        numberOfCommandQueuesForDevice = { (deviceId: Long, capabilities: CLCapabilities) =>
          1
        }
      ))

  "forward pass of matrix multiplication" in {

    configure
      .flatMap { hyperparameters0 =>
        val hyperparameters = hyperparameters0
        import hyperparameters._
        import hyperparameters.implicits._
        info("Given I have a 3x3 matrix of (0f, 1f, 2f, 4f, 7f, 10f, 13f, 15f, 17f) and a 3x1 matrix of (3f, 13f, 19f)")

        deviceBufferOf(0f, 1f, 2f, 4f, 7f, 10f, 13f, 15f, 17f)
          .flatMap { trainingQuestions =>
            deviceBufferOf(3f, 13f, 19f).flatMap { expectedAnswers =>
              info("When I ask to matrixMultiply two matrix")
              val matrixCalculate: Do[Tape[DeviceBuffer[Float], DeviceBuffer[Float]]] =
                hyperparameters.matrixMultiply(trainingQuestions, expectedAnswers, 3).forward
              info("Then I should have a 3x1 matrix of (51f, 293f, 557f)")

              matrixCalculate.flatMap {
                case (tape @ Tape(data, backward)) =>
                  data.toHostBuffer.map { buffer =>
                    val expected = Seq(51f, 293f, 557f)

                    0.until(buffer.capacity).map(buffer.get) should be(expected)

                  }

              }
            }
          }
      }
      .run
      .toScalaFuture

  }

//  "backward pass of matrix multiplication" in {
//    configure
//      .flatMap { hyperparameters0 =>
//        val hyperparameters = hyperparameters0
//        import hyperparameters._
//        import hyperparameters.implicits._
//        info("Given I have a 3x3 matrix of (0f, 1f, 2f, 4f, 7f, 10f, 13f, 15f, 17f) and a 3x1 matrix of (3f, 13f, 19f)")
//
//        deviceBufferOf(0f, 1f, 2f, 4f, 7f, 10f, 13f, 15f, 17f)
//          .flatMap { trainingQuestions =>
//            deviceBufferOf(3f, 13f, 19f).flatMap { expectedAnswers =>
//              info("When I ask to matrixMultiply two matrix")
//              val matrixCalculate: Do[Tape[DeviceBuffer[Float], DeviceBuffer[Float]]] =
//                hyperparameters.matrixMultiply(trainingQuestions, expectedAnswers, 3).forward
//              info("Then I should have a 3x1 matrix of (51f, 293f, 557f)")
//
//
//              }
//            }
//          }
//      }
//      .run
//      .toScalaFuture
//  }

}
