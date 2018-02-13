//package com.thoughtworks.deeplearning.plugins
//
//import java.io.FileInputStream
//import java.nio.ByteBuffer
//import java.util.concurrent.Executors
//
//import com.thoughtworks.compute.{Memory, OpenCL}
//import com.thoughtworks.continuation.{ParallelContinuation, UnitContinuation}
//import com.thoughtworks.deeplearning.DeepLearning
//import com.thoughtworks.deeplearning.DeepLearning.Tape
//import com.thoughtworks.deeplearning.plugins.Layers.ToLayer
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
//import shapeless.Witness
//
//import scala.concurrent.Await
//import scala.concurrent.duration.Duration
//import scalaz.std.stream._
//import scala.concurrent.ExecutionContext
//import scala.util.Random
//import scalaz.Tags.Parallel
//import scalaz.{Apply, Semigroup, Tags}
//
//object ExitCode134Debug {
//
//  trait FloatDeviceBufferLearningRate extends FloatDeviceBufferWeights {
//
//    implicit private def witnessThis: Witness.Aux[this.type] = Witness.mkWitness[this.type](this)
//
//    val learningRate: Float
//
//    private lazy val multiplyFloat: Program = {
//      val program = createProgramWithSource(
//        Seq("""
//        kernel void fill_value(global float* restrict operand0, float operand1, global float* restrict output) {
//          const size_t i = get_global_id(0);
//          output[i] = operand0[i] * operand1;
//        }
//      """)
//      )
//
//      program.build()
//      program
//    }
//
//    trait FloatDeviceBufferOptimizerApi extends super.FloatDeviceBufferOptimizerApi {
//      this: FloatDeviceBufferOptimizer =>
//      override def delta: Do[DeviceBuffer[Float]] = {
//        super.delta.flatMap { delta =>
//          val length = delta.length
//          allocateBuffer[Float](length).flatMap { newDelta: DeviceBuffer[Float] =>
//            Do.monadicCloseable(multiplyFloat.createFirstKernel())
//              .flatMap { kernel =>
//                kernel(0) = delta
//                kernel(1) = learningRate
//                kernel(2) = newDelta
//                val doEvent: Do[Event] = kernel.enqueue(length)
//                doEvent.flatMap { event =>
//                  Do.garbageCollected(event.waitForComplete())
//                }
//              }
//              .intransitiveMap { _: Unit =>
//                newDelta
//              }
//          }
//        }
//      }
//    }
//
//    type FloatDeviceBufferOptimizer <: FloatDeviceBufferOptimizerApi with Optimizer
//  }
//
//  private[ExitCode134Debug] trait DeviceBufferOf extends OpenCL {
//    def deviceBufferOf[Element](elements: Element*)(implicit memory: Memory[Element]): Do[DeviceBuffer[Element]] = {
//      val hostBuffer: memory.HostBuffer = memory.allocate(elements.length)
//      // TODO: optimize the performance
//      for ((element, i) <- elements.view.zipWithIndex) {
//        memory.put(hostBuffer, i, element)
//      }
//      allocateBufferFrom[Element, memory.HostBuffer](hostBuffer)(memory)
//    }
//  }
//
//  def main(args: Array[String]): Unit = {
//    new ExitCode134Debug().test.run.blockingAwait
//  }
//
//  implicit val executionContext = ExecutionContext.fromExecutor(Executors.newSingleThreadExecutor())
//}
//
//class ExitCode134Debug {
//  import ExitCode134Debug._
//
//  val random = new Random
//
//  // Workaround for https://github.com/milessabin/shapeless/issues/755
//  implicit private def witnessThis: Witness.Aux[this.type] = Witness.mkWitness(this)
//
//  def test: Do[Stream[Float]] = {
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
//      Logging with FloatLayers with OpenCLBufferLiterals with FloatTraining with ImplicitsSingleton with OpenCL.UseFirstDevice with OpenCL.UseFirstPlatform with OpenCL.CommandQueuePool with DeviceBufferWeights with FloatDeviceBufferWeights with DeviceBufferLayers with FloatDeviceBufferLayers with DeviceBufferOf with FloatDeviceBufferLearningRate]
//      .newInstance(
//        handleOpenCLNotification = handleOpenCLNotification,
//        numberOfCommandQueuesForDevice = { (deviceId: Long, capabilities: CLCapabilities) =>
//          1
//        },
//        learningRate = 0.0001f
//      ))
//
//    doHyperparameters.flatMap { hyperparameters0 =>
//      val hyperparameters = hyperparameters0
//      import hyperparameters._
//      import hyperparameters.implicits._
//      def randn(length: Int)(implicit memory: Memory[Float]): Do[DeviceBuffer[Float]] = {
//        val randomSeq = Seq.fill(length)(random.nextFloat())
//        deviceBufferOf[Float](randomSeq: _*)
//      }
//
//      val initialValueOfRobotWeights: Do[DeviceBuffer[Float]] = randn(3)
//
//      initialValueOfRobotWeights.flatMap { initialValueOfRobotWeight =>
//        val robotWeight: FloatDeviceBufferWeight = FloatDeviceBufferWeight(initialValueOfRobotWeight)
//
//        deviceBufferOf(0f, 1f, 2f, 4f, 7f, 10f, 13f, 15f, 17f).flatMap { trainingQuestions =>
//          def squareLoss(questions: DeviceBuffer[Float]): FloatLayer = {
//            val iqValue = matrixMultiply(questions, robotWeight, 3) //3 is matrix0Column
//            val meanValue = mean(multiply(iqValue, iqValue))
//            println(s"difference=$meanValue")
//            meanValue
//          }
//
//          val TotalIterations = 500
//
//          @monadic[Future]
//          def train: Future[Stream[Float]] = {
//            for (iteration <- (0 until TotalIterations).toStream) yield {
//              val loss = squareLoss(trainingQuestions).train.each
//              println(s"iteration=$iteration loss=$loss")
//              loss
//            }
//          }
//
//          Do.garbageCollected(train)
//        }
//      }
//
//    }
//  }
//
//}
