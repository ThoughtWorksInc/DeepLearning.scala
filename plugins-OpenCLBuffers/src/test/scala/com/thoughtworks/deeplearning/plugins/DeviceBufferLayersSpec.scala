package com.thoughtworks.deeplearning.plugins

import java.nio.{ByteBuffer, FloatBuffer}

import com.thoughtworks.compute.{Memory, OpenCL}
import com.thoughtworks.continuation.UnitContinuation
import com.thoughtworks.future._
import com.thoughtworks.feature.Factory
import com.thoughtworks.feature.mixins.ImplicitsSingleton
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
      DeviceBufferOf with FloatLayers with OpenCLBufferLiterals with ImplicitsSingleton with OpenCL.UseFirstDevice with OpenCL.UseFirstPlatform with OpenCL.CommandQueuePool with DeviceBufferWeights with FloatDeviceBufferWeights with DeviceBufferLayers with FloatDeviceBufferLayers]
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
              val matrixCalculate /*: Do[Tape[DeviceBuffer[Float], DeviceBufferWeights[Float]]]*/ =
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

  "backward pass of matrix multiplication and right can train" in {
    configure
      .flatMap { hyperparameters0 =>
        val hyperparameters = hyperparameters0
        import hyperparameters._
        import hyperparameters.implicits._
        info("Given I have a 3x3 matrix of (0f, 1f, 2f, 4f, 7f, 10f, 13f, 15f, 17f) and a 3x1 matrix of (3f, 13f, 19f)")

        val weight: Do[FloatDeviceBufferWeight] = deviceBufferOf(3f, 13f, 19f).map {
          expectedAnswers: DeviceBuffer[Float] =>
            FloatDeviceBufferWeight(expectedAnswers)
        }

        deviceBufferOf(0f, 1f, 2f, 4f, 7f, 10f, 13f, 15f, 17f)
          .flatMap { trainingQuestions =>
            weight.flatMap { weights =>
              info("When I ask to matrixMultiply two matrix")
              val matrixCalculate =
                hyperparameters.matrixMultiply(trainingQuestions, weights /*right train*/, 3).forward
              matrixCalculate
                .intransitiveFlatMap { tape =>
                  val backwardBuffer = deviceBufferOf(3f, 13f, 19f)
                  Do.garbageCollected(tape.backward(backwardBuffer))
                }
                .flatMap { _: Unit =>
                  info("Then I should have a 3x1 matrix of (-296f,-366f,-440f)")
                  weights.data.toHostBuffer.map { buffer =>
                    val expected = Seq(-296f, -366f, -440f)

                    0.until(buffer.capacity).map(buffer.get) should be(expected)

                  }
                }

            }

          }
      }
      .run
      .toScalaFuture
  }

  "backward pass of matrix multiplication and left can train" in {
    configure
      .flatMap { hyperparameters0 =>
        val hyperparameters = hyperparameters0
        import hyperparameters._
        import hyperparameters.implicits._
        info("Given I have a 3x3 matrix of (0f, 1f, 2f, 4f, 7f, 10f, 13f, 15f, 17f) and a 3x1 matrix of (3f, 13f, 19f)")

        val weight: Do[FloatDeviceBufferWeight] = deviceBufferOf(0f, 1f, 2f, 4f, 7f, 10f, 13f, 15f, 17f).map {
          expectedAnswers: DeviceBuffer[Float] =>
            FloatDeviceBufferWeight(expectedAnswers)
        }
        deviceBufferOf(3f, 13f, 19f)
          .flatMap { trainingQuestions =>
            weight.flatMap { weights =>
              info("When I ask to matrixMultiply two matrix")
              val matrixCalculate =
                hyperparameters.matrixMultiply(weights /*left*/, trainingQuestions, 3).forward
              matrixCalculate
                .intransitiveFlatMap { tape =>
                  val backwardBuffer = deviceBufferOf(3f, 13f, 19f)
                  Do.garbageCollected(tape.backward(backwardBuffer))
                }
                .flatMap { _: Unit =>
                  info("Then I should have a 3x3 matrix of (-9f, -38f, -55f, -35f, -162f, -237f, -44f, -232f, -344f)")
                  weights.data.toHostBuffer.map { buffer =>
                    val expected = Seq(-9f, -38f, -55f, -35f, -162f, -237f, -44f, -232f, -344f)

                    0.until(buffer.capacity).map(buffer.get) should be(expected)

                  }
                }

            }

          }
      }
      .run
      .toScalaFuture
  }

  "1x1 backward pass of matrix multiplication" in {
    configure
      .flatMap { hyperparameters0 =>
        val hyperparameters = hyperparameters0
        import hyperparameters._
        import hyperparameters.implicits._
        info("Given I have a 1x1 matrix of (10f) and a 1x1 weight matrix of (1000f)")

        val weight: Do[FloatDeviceBufferWeight] = deviceBufferOf(1000f).map { expectedAnswers: DeviceBuffer[Float] =>
          FloatDeviceBufferWeight(expectedAnswers)
        }

        deviceBufferOf(10f)
          .flatMap { trainingQuestions =>
            weight.flatMap { weights =>
              info("When I ask to matrixMultiply two matrix")
              val matrixCalculate =
                hyperparameters.matrixMultiply(trainingQuestions, weights, 1).forward
              matrixCalculate
                .intransitiveFlatMap { tape =>
                  val backwardBuffer = deviceBufferOf(2f)
                  Do.garbageCollected(tape.backward(backwardBuffer))
                }
                .flatMap { _: Unit =>
                  info("Then I should have a 1x1 matrix of (980f)")
                  weights.data.toHostBuffer.map { buffer =>
                    val expected = Seq(980f)

                    0.until(buffer.capacity).map(buffer.get) should be(expected)

                  }
                }

            }

          }
      }
      .run
      .toScalaFuture

  }

  "1x1 forward pass of multiply" in {
    configure
      .flatMap { hyperparameters0 =>
        val hyperparameters = hyperparameters0
        import hyperparameters._
        import hyperparameters.implicits._
        info("Given I have a 1x1 matrix of (10f) and a 1x1 matrix of (10f)")

        deviceBufferOf(10f)
          .flatMap { trainingQuestions =>
            deviceBufferOf(10f).flatMap { weights =>
              info("When I ask to matrixMultiply two matrix")
              val matrixCalculate =
                hyperparameters.multiply(trainingQuestions, weights).forward
              matrixCalculate.flatMap {
                case (tape @ Tape(data, backward)) =>
                  data.toHostBuffer.map { buffer =>
                    info("Then I should have a 1x1 matrix of (100f)")
                    val expected = Seq(100f)

                    0.until(buffer.capacity).map(buffer.get) should be(expected)

                  }
              }
            }

          }

      }
      .run
      .toScalaFuture

  }

  "2x1 backward pass of multiply" in {
    configure
      .flatMap { hyperparameters0 =>
        val hyperparameters = hyperparameters0
        import hyperparameters._
        import hyperparameters.implicits._
        info("Given I have a 2x1 matrix of (10f, 20f) and a 2x1 weight matrix of (1000f, 2000f)")

        val weight: Do[FloatDeviceBufferWeight] = deviceBufferOf(1000f, 2000f).map {
          expectedAnswers: DeviceBuffer[Float] =>
            FloatDeviceBufferWeight(expectedAnswers)
        }

        deviceBufferOf(10f, 20f)
          .flatMap { trainingQuestions =>
            weight.flatMap { weights =>
              info("When I ask to matrixMultiply two matrix")
              val matrixCalculate = hyperparameters.multiply(trainingQuestions, weights).forward
              matrixCalculate
                .intransitiveFlatMap { tape =>
                  val backwardBuffer = deviceBufferOf(2f, 4f)
                  Do.garbageCollected(tape.backward(backwardBuffer))
                }
                .flatMap { _: Unit =>
                  info("Then I should have a 1x1 matrix of (980f, 1920f)")
                  weights.data.toHostBuffer.map { buffer =>
                    val expected = Seq(980f, 1920f)

                    0.until(buffer.capacity).map(buffer.get) should be(expected)

                  }
                }

            }

          }
      }
      .run
      .toScalaFuture

  }

  "3x1 backward pass of multiply" in {
    configure
      .flatMap { hyperparameters0 =>
        val hyperparameters = hyperparameters0
        import hyperparameters._
        import hyperparameters.implicits._
        info("Given I have a 3x1 matrix of (10f, 20f, 30f)")

        deviceBufferOf(10f, 20f, 30f)
          .flatMap { trainingQuestions =>
            info("When I ask to matrixMultiply one matrix")
            val matrixCalculate = hyperparameters.mean(trainingQuestions).forward
            matrixCalculate
              .flatMap {
                case (tape @ Tape(data, backward)) =>
                  info("Then I should have a (20f)")
                  Do.delay(data should be(20f))

              }

          }
      }
      .run
      .toScalaFuture

  }

}
