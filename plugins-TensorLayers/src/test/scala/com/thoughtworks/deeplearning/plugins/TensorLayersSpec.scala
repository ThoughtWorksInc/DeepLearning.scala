package com.thoughtworks.deeplearning.plugins

import org.scalatest.{AsyncFreeSpec, Matchers}
import com.thoughtworks.feature.mixins.ImplicitsSingleton
import com.thoughtworks.future._
import com.thoughtworks.raii.asynchronous._
import com.thoughtworks.feature.{Factory, ImplicitApply}
import scalaz.syntax.all._
import com.thoughtworks.compute.{OpenCL, Tensors}
import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.plugins._
import com.typesafe.scalalogging.StrictLogging

/**
  * @author 杨博 (Yang Bo)
  */
class TensorLayersSpec extends AsyncFreeSpec with Matchers {

  "xxx" in {

    Do.monadicCloseable {
        Factory[
          TensorWeights with TensorLayers with Operators with StrictLogging with Tensors.UnsafeMathOptimizations with OpenCL.LogContextNotification with OpenCL.GlobalExecutionContext with OpenCL.CommandQueuePool with OpenCL.UseAllCpuDevices with OpenCL.DontReleaseEventTooEarly with OpenCL.SynchronizedCreatingKernel with OpenCL.HandleEventInExecutionContextForIntelAndAMDPlatform with Tensors.WangHashingRandomNumberGenerator with ImplicitsSingleton]
          .newInstance(numberOfCommandQueuesPerDevice = 5)
      }
      .flatMap { hyperparameters =>
        import hyperparameters._, implicits._
        TensorWeight.allocate(Tensor(Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))).flatMap { weight =>
          def plus = weight + weight

          plus.forward.flatMap { tape =>
            tape.data.toString should be("[[2.0,4.0],[6.0,8.0]]")
            Do.garbageCollected(tape.backward(Do.now(Tensor.scalar(1.0f)))).map { _: Unit =>
              tape.data.toString should be("[[2.0,4.0],[6.0,8.0]]")

              succeed

            }
          }

//          Do.garbageCollected(plus.train.map { y =>
//            y.toString should be("[[2.0,4.0],[6.0,8.0]]")
////                succeed
//          }.run)
//              .map { _: Any =>
////                weight.data.toString should be("[[1.0,2.0],[3.0,4.0]]")
//                succeed
//              })

        // FIXME: exception thrown here is ignored
//          weight.data.toString should be("[[1.0,2.0],[3.0,4.0]]")
        }
      }
      .run
      .toScalaFuture
  }

}
