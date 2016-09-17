package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.Differentiable.Batch
import com.thoughtworks.deepLearning.Differentiable.DifferentiableArray2D.Array2DLiteral
import com.thoughtworks.deepLearning.Dsl.DslFunction
import org.nd4j.linalg.api.ndarray.INDArray
import org.scalatest.{FreeSpec, Matchers}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DeepLearningSpec extends FreeSpec with Matchers {

  implicit def learningRate = new Differentiable.LearningRate {
    def apply() = 0.03
  }
  "XOR" in {

    final case class PredictXor[D <: Dsl](dsl: D) extends DslFunction with DeepLearning {

      import dsl._

      override type In = Array2D
      override type Out = Array2D
      //
      //      private def loss(likelihood: Array2D, expectedLabels: Array2D) = {
      //        -(log(likelihood) * expectedLabels).reduceSum
      //      }

      override def apply(in: In): Out = {
        sigmoid(fullyConnectedThenRelu(
          (0 until 10).foldLeft(fullyConnectedThenRelu(in, 2, 50)) { (hiddenLayer, _) =>
            fullyConnectedThenRelu(hiddenLayer, 50, 50)
          },
          50,
          2
        ))

      }

    }


    val predictor = Differentiable.fromDsl[PredictXor]
  }

}
