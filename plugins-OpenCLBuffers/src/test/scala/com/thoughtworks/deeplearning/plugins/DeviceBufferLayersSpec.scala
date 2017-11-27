package com.thoughtworks.deeplearning.plugins

import java.nio.{ByteBuffer, FloatBuffer}

import com.thoughtworks.compute.OpenCL.DeviceBuffer
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
  import DeviceBufferLayersSpec.handleOpenCLNotification
//  Feature: User trades stocks
//  Scenario: User requests a sell before close of trading
//    Given I have 100 shares of MSFT stock
//    And I have 150 shares of APPL stock
//    And the time is before close of trading
//
//  When I ask to sell 20 shares of MSFT stock
//
//  Then I should have 80 shares of MSFT stock
//  And I should have 150 shares of APPL stock
//  And a sell order for 20 shares of MSFT stock should have been executed
//  info("Given I created some configuration of hyperparameters")
  private def configure =
    Do.monadicCloseable(Factory[
      FloatLayers with OpenCLBufferLiterals with FloatTraining with ImplicitsSingleton with OpenCL.UseAllDevices with OpenCL.UseFirstPlatform with OpenCL.CommandQueuePool with DeviceBufferWeights with FloatDeviceBufferWeights with DeviceBufferLayers with FloatDeviceBufferLayers with NormalDistributionRandom]
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
        import hyperparameters.implicits._
        import hyperparameters.DeviceBufferWeight
        import hyperparameters.DeviceBuffer
        import hyperparameters.DeviceBufferLayer
        import hyperparameters.FloatDeviceBufferWeight
        import hyperparameters.FloatDeviceBufferLayer
        info("Given I have a 3x3 matrix of (0f, 1f, 2f, 4f, 7f, 10f, 13f, 15f, 17f) and a 3x1 matrix of (3f, 13f, 19f)")
        def deviceBufferOf[Element](elements: Element*)(implicit memory: Memory[Element]): Do[DeviceBuffer[Element]] = {
          val hostBuffer: memory.HostBuffer = memory.allocate(elements.length)
          // TODO: optimize the performance
          for ((element, i) <- elements.view.zipWithIndex) {
            memory.put(hostBuffer, i, element)
          }
          hyperparameters.allocateBufferFrom[Element, memory.HostBuffer](hostBuffer)(memory)
        }

        deviceBufferOf(0f, 1f, 2f, 4f, 7f, 10f, 13f, 15f, 17f)
          .flatMap { trainingQuestions =>
            deviceBufferOf(3f, 13f, 19f).flatMap { expectedAnswers =>
              info("When I ask to matrixMultiply two matrix")
              val matrixCalculate: Do[Tape[DeviceBuffer[Float], DeviceBuffer[Float]]] =
                hyperparameters.matrixMultiply(trainingQuestions, expectedAnswers, 3).forward
              info("Then I should have a 3x1 matrix of (51f, 293f, 557f)")

//              martixCalculate.map { tape =>
//                tape.data.toHostBuffer.flatMap { buffer: FloatBuffer =>
//                  Do.now(buffer.array() should be(Array[Float](51f, 293f, 557f))): Do[Assertion]
//                }

              matrixCalculate.flatMap {
                case (tape @ Tape(data, backward)) =>
                  data.toHostBuffer.map { buffer =>
                    val expected = Seq(51f, 293f, 557f)

                    0.until(buffer.capacity).map(buffer.get) should be(expected)

                  }

              }: Do[Assertion]
            }
          }
      }
      .run
      .toScalaFuture: scala.concurrent.Future[Assertion]

  }
//  matrix([[  51.],
//  [ 293.],
//  [ 557.]])
  /*
  com.thoughtworks.raii.asynchronous.asynchronousDoMonadErrorInstances.bind(fa.forward) { a =>
    ???
  }

  import com.thoughtworks.raii.asynchronous._
  Monad[Do].bind(fa.forward) { a =>
    ???
  }

//  import scalaz.syntax.all._
  import scalaz.syntax.monad._
  fa.forward.flatMap { a =>
    ???
  }

   * */

//  def test = {

//
//    doHyperparameters.flatMap { hyperparameters0 =>
//      val hyperparameters = hyperparameters0
//      import hyperparameters.implicits._
//      import hyperparameters.DeviceBufferWeight
//      import hyperparameters.DeviceBuffer
//      import hyperparameters.DeviceBufferLayer
//      import hyperparameters.FloatDeviceBufferWeight
//      import hyperparameters.FloatDeviceBufferLayer
//
//      def deviceBufferOf[Element](elements: Element*)(implicit memory: Memory[Element]): Do[DeviceBuffer[Element]] = {
//        val hostBuffer: memory.HostBuffer = memory.allocate(elements.length)
//        // TODO: optimize the performance
//        for ((element, i) <- elements.view.zipWithIndex) {
//          memory.put(hostBuffer, i, element)
//        }
//        hyperparameters.allocateBufferFrom[Element, memory.HostBuffer](hostBuffer)(memory)
//      }
//
//      deviceBufferOf(0f, 1f, 2f, 4f, 7f, 10f, 13f, 15f, 17f).flatMap { trainingQuestions =>
//        deviceBufferOf(3f, 13f, 19f).flatMap { expectedAnswers =>
//          val martixCalculate: Do[Tape[DeviceBuffer[Float], DeviceBuffer[Float]]] =
//            hyperparameters.matrixMultiply(trainingQuestions, expectedAnswers, 3).forward
//          martixCalculate.flatMap {
//            case (tape @ Tape(data, backward)) =>
//              data.toHostBuffer.flatMap { floatBuffer =>
//                floatBuffer.toString()
//
//              }
//          }
//
//        }
//      }
//    }
//  }

}
