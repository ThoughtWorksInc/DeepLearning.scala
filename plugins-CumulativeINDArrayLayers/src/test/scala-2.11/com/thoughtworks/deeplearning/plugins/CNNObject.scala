package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.each.Monadic.monadic
import com.thoughtworks.feature.ImplicitApply
import com.thoughtworks.raii.asynchronous.Do
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.IsMax
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.util.ArrayUtil
import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.plugins._
import com.thoughtworks.each.Monadic.{monadic, _}
import com.thoughtworks.feature.{Factory, ImplicitApply}
import com.thoughtworks.raii.asynchronous._
import org.nd4j.linalg.api.ops.impl.transforms.IsMax
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.util.ArrayUtil
import org.nd4s.Implicits._

import collection.immutable.IndexedSeq
import scala.io.Source
import scala.concurrent.ExecutionContext.Implicits.global
import scalaz.std.iterable._
import scalaz.syntax.all._
import com.thoughtworks.future._

import scalaz.std.stream._
import scala.concurrent.Await
import scala.concurrent.duration.Duration
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import com.thoughtworks.feature.Factory

object CNNObject {
  trait CNNs extends INDArrayLayers with ImplicitsSingleton with Training with Operators {

    trait ImplicitsApi
        extends super[INDArrayLayers].ImplicitsApi
        with super[Training].ImplicitsApi
        with super[Operators].ImplicitsApi

    type Implicits <: ImplicitsApi

    private def toArray(tuple2: (Int, Int)): Array[Int] = {
      val (one, two) = tuple2
      Array(one, two)
    }

    def im2col[Operand0, Out <: INDArrayLayer](operand0: Operand0,
                                               kernel: (Int, Int),
                                               stride: (Int, Int),
                                               padding: (Int, Int))(
        implicit deepLearning: DeepLearning.Aux[Operand0, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]): Out = {
      INDArrayLayer.unary(operand0) { data0: INDArray =>
        val shape0 = data0.shape
        val strideArray = toArray(stride)
        val paddingArray = toArray(padding)
        val outputData =
          Convolution.im2col(data0, toArray(kernel), strideArray, paddingArray)
        val delta0 = { outputDelta: INDArray =>
          Convolution.col2im(outputDelta, strideArray, paddingArray, shape0(2), shape0(3))
        }
        (outputData, delta0)
      }
    }

    @inline
    def conv2d[Input, Weight, Bias, Out <: INDArrayLayer](input: Input,
                                                          weight: Weight,
                                                          bias: Bias,
                                                          kernel: (Int, Int),
                                                          stride: (Int, Int),
                                                          padding: (Int, Int))(
        implicit inputDeepLearning: DeepLearning.Aux[Input, INDArray, INDArray],
        weightDeepLearning: DeepLearning.Aux[Weight, INDArray, INDArray],
        biasDeepLearning: DeepLearning.Aux[Bias, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]): Out = {
      import implicits._
      INDArrayLayer(monadic[Do] {
        val inputShape = input.forward.each.data.shape
        val numberOfImages = inputShape(0)
        val depth = inputShape(1)
        val height = inputShape(2)
        val width = inputShape(3)
        val numberOfKernels = weight.forward.each.data.shape.head
        val col = im2col(input, kernel, stride, padding)
        val permutedCol = col.permute(0, 4, 5, 1, 2, 3)
        val depthKernelKernel = depth * kernel._1 * kernel._2
        val operandCol2d = permutedCol.reshape(numberOfImages * height * width, depthKernelKernel)
        val reshapedWeight = weight.reshape(numberOfKernels, depthKernelKernel)
        val permutedWeight = reshapedWeight.permute(1, 0)
        val dotResult = operandCol2d dot permutedWeight
        val plusResult = dotResult + bias
        val reshapeResult =
          plusResult.reshape(numberOfImages, height, width, numberOfKernels)
        reshapeResult.permute(0, 3, 1, 2).forward.each
      })
    }

    @inline
    def maxPool[Operand0, Out <: INDArrayLayer](operand0: Operand0, poolSize: (Int, Int))(
        implicit deepLearning: DeepLearning.Aux[Operand0, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]): Out = {
      INDArrayLayer.unary(operand0) { data0: INDArray =>
        val shape0 = data0.shape
        val kernelAndStrideSize: Array[Int] = toArray(poolSize)
        val preMaxPool: INDArray =
          Convolution
            .im2col(data0, kernelAndStrideSize, kernelAndStrideSize, Array(0, 0))
            .permute(0, 1, 4, 5, 2, 3)
        val preShape: Seq[Int] = preMaxPool.shape().toSeq
        val lastDimensionSize: Int = preShape.takeRight(2).product
        val reshapedPreMaxPool: INDArray = preMaxPool
          .reshape(preShape.take(preShape.length - 2) :+ lastDimensionSize: _*)
        val outputData = reshapedPreMaxPool.max(4)
        val delta0 = { outputDelta: INDArray =>
          val a = reshapedPreMaxPool
          val upStreamDup = a.dup()
          val rows = ArrayUtil.prod(a.length())

          val isMax: INDArray = Nd4j.getExecutioner
            .execAndReturn(new IsMax(upStreamDup, 4))
            .reshape(preShape
              .take(preShape.length - 2) :+ poolSize._2 :+ poolSize._1: _*)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape('c', rows, 1)

          val outputDelta1d = {
            outputDelta
              .repeat(-1, poolSize._1)
              .permute(1, 0, 3, 2)
              .repeat(-1, poolSize._2)
              .permute(1, 0, 3, 2)
              .reshape('c', shape0.product, 1)
          }

          isMax
            .muliColumnVector(outputDelta1d)
            .reshape(shape0: _*)
        }
        (outputData, delta0)
      }
    }
  }

  trait LearningRate extends INDArrayWeights {
    val learningRate: Double

    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi {
      this: INDArrayOptimizer =>
      override def delta: INDArray = super.delta mul learningRate
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi with Optimizer
  }

  trait Adagrad extends INDArrayWeights {
    val eps: Double

    trait INDArrayWeightApi extends super.INDArrayWeightApi {
      this: INDArrayWeight =>
      var cache: Option[INDArray] = None
    }

    override type INDArrayWeight <: INDArrayWeightApi with Weight

    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi {
      this: INDArrayOptimizer =>
      private lazy val deltaLazy: INDArray = {
        import org.nd4s.Implicits._
        import weight._
        val delta0 = super.delta
        cache = Some(cache.getOrElse(Nd4j.zeros(delta0.shape: _*)) + delta0 * delta0)
        delta0 / (Transforms.sqrt(cache.get) + eps)
      }
      override def delta = deltaLazy
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi with Optimizer
  }
}
