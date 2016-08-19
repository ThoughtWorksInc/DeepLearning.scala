package com.thoughtworks

import com.thoughtworks.DeepLearning.{Array2D, DifferentiableINDArray, PointfreeDeepLearning}

import scala.language.existentials
import scala.language.higherKinds
import Predef.{any2stringadd => _, _}
import PointfreeDeepLearning.ops._
import cats.{Applicative, Eval}
import com.thoughtworks.Differentiable.DifferentiableFunction
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import org.scalatest.{FreeSpec, Matchers}
import org.nd4s.Implicits._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DeepLearningSpec extends FreeSpec with Matchers {

  final def addDoubleDouble[F[_] : PointfreeDeepLearning](left: F[Double], right: F[Double]): F[Double] = {
    left + right
  }

  final def addDoubleArray[F[_] : PointfreeDeepLearning](left: F[Double], right: F[Array2D]): F[Array2D] = {
    left + right
  }

  final def addArrayDouble[F[_] : PointfreeDeepLearning](left: F[Array2D], right: F[Double]): F[Array2D] = {
    left + right
  }

  final def addArrayArray[F[_] : PointfreeDeepLearning](left: F[Array2D], right: F[Array2D]): F[Array2D] = {
    left + right
  }

  "XOR" in {

    def xorNetwork[F[_] : PointfreeDeepLearning](input: F[Array2D]): F[Array2D] = {
      input.fullyConnectedThenRelu(2, 20).fullyConnectedThenRelu(20, 20).fullyConnectedThenRelu(20, 20).fullyConnectedThenRelu(20, 20).fullyConnectedThenRelu(20, 2).sigmoid
    }


    val myNetwork: Differentiable[Array2D => Array2D] = xorNetwork[Lambda[A => Differentiable[Array2D => A]]](PointfreeDeepLearning[Differentiable].id[Array2D])

    val forwardPass = myNetwork.asInstanceOf[DifferentiableFunction[Array2D, Array2D,_,_]].forward(PointfreeDeepLearning[Differentiable].liftArray2D(
      Array(
        Array(0.0, 0.0),
        Array(0.0, 1.0),
        Array(1.0, 0.0),
        Array(1.0, 1.0)
      )
    ): Differentiable.Aux[Array2D, _, _])

    def lossFunction(output: Differentiable[Array2D]): Eval[Option[INDArray]] = {
      val labels = Array(
        Array(1.0, 0.0),
        Array(0.0, 1.0),
        Array(0.0, 1.0),
        Array(1.0, 0.0)
      ).toNDArray

      //      var
      output.asInstanceOf[DifferentiableINDArray].data.map { scores: INDArray =>
        val l = Transforms.exp(scores * labels) / scores.sumT
        //        println(l)
        val deltaLoss = (labels rsub 1) / (scores rsub 1) - labels / scores
        Some(deltaLoss): Option[INDArray]
      }
    }

    val loss = lossFunction(forwardPass.output)

    val backwardPass = forwardPass.backward(loss.asInstanceOf[Eval[forwardPass.OutputDelta]])

    val learningRate = 0.003
    val newWeight = Applicative[Eval].map3(myNetwork.patch, myNetwork.data, backwardPass.deltaWeight) { (patch, weight, deltaWeight) =>
      patch(weight, deltaWeight.asInstanceOf[myNetwork.Delta], learningRate)
    }.value

    println(newWeight)

  }

}
