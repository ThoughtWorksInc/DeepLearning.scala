package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.compute.{OpenCL, Tensors}
import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.future._
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.feature.{Factory, ImplicitApply}
import com.thoughtworks.feature.mixins.ImplicitsSingleton
import com.thoughtworks.raii.asynchronous._
import com.typesafe.scalalogging.StrictLogging
import org.scalatest._
import scalaz.Applicative
import scalaz.syntax.all._
import scalaz.std.list._
import scalaz.std.iterable._

private object CumulativeTensorLayersSpec {

  trait CNNs extends ImplicitsSingleton with Operators with CumulativeTensorLayers with TensorWeights {
    type Implicits <: super[Operators].ImplicitsApi with super[CumulativeTensorLayers].ImplicitsApi with super[
      TensorWeights].ImplicitsApi
    import implicits._

    def convolute[Out <: TensorLayer, I: DeepLearningTensor, W: DeepLearningTensor, B: DeepLearningTensor](
        input: I /* batchSize × height × width × depth */,
        weight: W /* kernelHeight × kernelWidth × depth × filterSize */,
        bias: B /* filterSize */ )(implicit
                                   implicitApply: ImplicitApply.Aux[tensorPartialApplyRawForward.Rest, Out]): Out = {
      TensorLayer(
        bias.split(0).flatMap { biasSeq =>
          weight.split(3).flatMap {
            khKwD =>
              List
                .tabulate(biasSeq.length) { o =>
                  khKwD(o).split(0).flatMap { kwD =>
                    kwD.indices.foldLeftM(biasSeq(o)) { (acc, y) =>
                      kwD(y).split(0).flatMap { d =>
                        d.indices.foldLeftM(acc) { (acc, x) =>
                          Applicative[Do].apply2(input.split(3), d(x).split(0)) { (channel, w) =>
                            w.indices.foldLeft(acc) { (acc, i) =>
                              acc + channel(i).translate(Array(0, y - kwD.length / 2, x - d.length / 2)) * w(i)
                            }
                          }
                        }
                      }
                    }
                  }
                }
                .sequence
                .flatMap { s =>
                  join(s).forward
                }
          }
        }
      )
    }
  }
}

/**
  * @author 杨博 (Yang Bo)
  */
final class CumulativeTensorLayersSpec extends AsyncFreeSpec with Matchers {
  import CumulativeTensorLayersSpec._

  "convolute" in {

    Do.monadicCloseable {
        Factory[
          Logging with TensorLiterals with TensorWeights with CNNs with Operators with StrictLogging with Tensors.UnsafeMathOptimizations with OpenCL.LogContextNotification with OpenCL.GlobalExecutionContext with OpenCL.CommandQueuePool with OpenCL.UseAllCpuDevices with OpenCL.DontReleaseEventTooEarly with OpenCL.SynchronizedCreatingKernel with OpenCL.HandleEventInExecutionContextForIntelAndAMDPlatform with Tensors.WangHashingRandomNumberGenerator with ImplicitsSingleton]
          .newInstance(numberOfCommandQueuesPerDevice = 5)
      }
      .flatMap { hyperparameters =>
        import hyperparameters._, implicits._
        TensorWeight
          .allocate {
            val weightArray = Array.ofDim[Float](3, 3, 3, 2) /* kernelHeight × kernelWidth × depth × filterSize */
            weightArray(1)(1)(0)(0) = 3.0f
            weightArray(1)(1)(0)(1) = 4.0f
            weightArray(0)(1)(0)(0) = 5.0f
            weightArray(2)(2)(0)(1) = 6.0f

            Tensor(weightArray)
          }
          .flatMap { weight =>
            TensorWeight
              .allocate {
                val biasArray = Array[Float](100000.0f, 200000.0f) /* filterSize */
                Tensor(biasArray)
              }
              .flatMap { bias =>
                val inputArray = Array.ofDim[Float](2, 4, 5, 3) /* batchSize × height × width × depth */

                inputArray(0)(0)(0)(0) = 1.0f
                inputArray(0)(1)(0)(0) = 10.0f
                inputArray(1)(0)(0)(0) = 100.0f
                val inputTensor = Tensor(inputArray)

                val outputLayer = convolute(
                  input = inputTensor,
                  weight = weight,
                  bias = bias
                )

                Do.garbageCollected(
                  outputLayer.train
                    .intransitiveFlatMap { outputTensor =>
                      Do.garbageCollected(outputTensor.flatArray).intransitiveMap {
                        a =>
                          val outputArray = a
                            .grouped(2)
                            .toArray
                            .grouped(5)
                            .toArray
                            .grouped(4)
                            .toArray

                          outputArray.length should be(2)

                          outputArray(0)(0)(0)(0) should be(100053.0f)
                          outputArray(0)(1)(1)(1) should be(200006.0f)
                          outputArray(1)(1)(1)(1) should be(200600.0f)
                          outputArray(0)(2)(1)(1) should be(200060.0f)
                          outputArray(0)(0)(0)(1) should be(200004.0f)
                          outputArray(0)(1)(0)(0) should be(100030.0f)
                          outputArray(1)(0)(0)(0) should be(100300.0f)
                      }
                    }
                    .run
                    .flatMap { _: Assertion =>
                      weight.data.flatArray.map { a =>
                        val weightArray = a.grouped(2).toArray.grouped(3).toArray.grouped(3).toArray
//                        weightArray(1)(1)(0)(0) should be(null)
                        succeed
                      }
                    })
              }
          }
      }
      .run
      .toScalaFuture

  }
}
