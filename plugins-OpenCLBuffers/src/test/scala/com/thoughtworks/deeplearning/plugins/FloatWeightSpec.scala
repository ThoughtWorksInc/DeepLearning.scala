//package com.thoughtworks.deeplearning.plugins
//
//import java.io.FileInputStream
//import java.nio.ByteBuffer
//
//import com.thoughtworks.compute.OpenCL.DeviceBuffer
//import com.thoughtworks.compute.{Memory, OpenCL}
//import com.thoughtworks.deeplearning.DeepLearning
//import com.thoughtworks.feature.Factory
//import com.thoughtworks.feature.mixins.ImplicitsSingleton
//import com.thoughtworks.future._
//import com.thoughtworks.raii.asynchronous._
//import org.lwjgl.opencl.CLCapabilities
//import org.scalatest.{FreeSpec, Matchers}
//
//import scalaz.syntax.all._
//import com.thoughtworks.each.Monadic.monadic
//import com.thoughtworks.each.Monadic._
//import com.thoughtworks.future._
//
//import scala.concurrent.Await
//import scala.concurrent.duration.Duration
//import scalaz.std.stream._
//import scala.concurrent.ExecutionContext.Implicits.global
//
//// INDArray -> OpenCLBuffer
//object FloatWeightSpec {
////  trait NormalDistributionRandom extends OpenCL {
////    def randn[Element](length: Int)(implicit memory: Memory[Element]): Do[DeviceBuffer[Element]] = ???
////  }
//
//  def main(args: Array[String]): Unit = {
//    new FloatWeightSpec().test.run.blockingAwait
//  }
//}
//final class FloatWeightSpec {
//
//  import FloatWeightSpec._
//
//  def test: Do[Stream[Float]] = {
//
//    import scala.util.Random
//    val random = new Random()
//
//    val handleOpenCLNotification = { (errorInfo: String, buffer: ByteBuffer) =>
//      if (buffer.remaining > 0) {
//        val hexText = for (i <- (buffer.position until buffer.limit).view) yield {
//          f"${buffer.get(i)}%02X"
//        }
//        Console.err.println(hexText.mkString(errorInfo, " ", ""))
//        Console.err.flush()
//      } else {
//        Console.err.println(errorInfo)
//        Console.err.flush()
//      }
//    }
//    val doHyperparameters = Do.monadicCloseable(Factory[
//      Logging with Names with FloatLayers with OpenCLBufferLiterals with FloatTraining with ImplicitsSingleton with OpenCL.UseFirstGpuDevice with OpenCL.UseFirstPlatform with OpenCL.CommandQueuePool with DeviceBufferWeights with FloatDeviceBufferWeights with DeviceBufferLayers with FloatDeviceBufferLayers]
//      .newInstance(
//        handleOpenCLNotification = handleOpenCLNotification,
//        numberOfCommandQueuesForDevice = { (deviceId: Long, capabilities: CLCapabilities) =>
//          1
//        }
//      ))
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
//      def randn(length: Int)(implicit memory: Memory[Float]): Do[DeviceBuffer[Float]] = {
//        val randomSeq = (0 until length).map(_ => 1000000f /*_ => random.nextFloat()*/ )
//        deviceBufferOf[Float](randomSeq: _*)
//      }
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
//      val initialValueOfRobotWeights: Do[DeviceBuffer[Float]] = randn(3)
//
//      initialValueOfRobotWeights.flatMap { initialValueOfRobotWeight =>
//        val robotWeight: FloatDeviceBufferWeight = FloatDeviceBufferWeight(initialValueOfRobotWeight)
//
//        deviceBufferOf(0f, 1f, 2f, 4f, 7f, 10f, 13f, 15f, 17f).flatMap { trainingQuestions =>
//          deviceBufferOf(3f, 13f, 19f).flatMap { expectedAnswers =>
//            def iqTestRobot(questions: hyperparameters.DeviceBuffer[Float]): FloatDeviceBufferLayer = {
//              hyperparameters.matrixMultiply(questions, robotWeight, 3) //3 is matrix0Columns
//            }
//
//            def squareLoss(questions: DeviceBuffer[Float],
//                           expectAnswer: DeviceBuffer[Float]): hyperparameters.FloatLayer = {
//              val difference = hyperparameters.subtract(iqTestRobot(questions), expectAnswer)
//              val loss = hyperparameters.multiply(difference, difference)
//              hyperparameters.mean(loss)
//
//            }
//
//            val TotalIterations = 500
//
//            @monadic[Future]
//            def train: Future[Stream[Float]] = {
//              for (iteration <- (0 until TotalIterations).toStream) yield {
//                val loss = squareLoss(trainingQuestions, expectedAnswers).train.each
//                hyperparameters.logger.info(s"iteration=$iteration loss=$loss")
//                loss
//              }
//            }
//
//            Do.garbageCollected(train)
//          }
//        }
//      }
//    }
//  }
//}
