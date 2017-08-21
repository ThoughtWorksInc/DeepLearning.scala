package com.thoughtworks.deeplearning
package plugins

import com.thoughtworks.deeplearning.scalatest.ThoughtworksFutureToScalaFuture
import com.thoughtworks.each.Monadic._
import com.thoughtworks.feature.{Factory, ImplicitApply}
import com.thoughtworks.raii.asynchronous._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.IsMax
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.util.ArrayUtil
import org.scalatest._
import org.nd4s.Implicits._
import com.thoughtworks.continuation._
import com.thoughtworks.feature.mixins.ImplicitsSingleton
import com.thoughtworks.future._

import scalaz.std.iterable._

object CumulativeINDArrayLayersSpec {

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
        val outputData = Convolution.im2col(data0, toArray(kernel), strideArray, paddingArray)
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
        val reshapeResult = plusResult.reshape(numberOfImages, height, width, numberOfKernels)
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
            .reshape(preShape.take(preShape.length - 2) :+ poolSize._2 :+ poolSize._1: _*)
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

  trait FixedLearningRate extends LearningRate {
    val fixedLearningRate: scala.Double
    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi { this: INDArrayOptimizer =>
      final def learningRate: scala.Double = fixedLearningRate
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi with Optimizer
  }

  trait LearningRate extends INDArrayWeights {
    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi { this: INDArrayOptimizer =>
      def learningRate: scala.Double
      abstract override def delta: INDArray = super.delta mul learningRate
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi with Optimizer
  }

  trait L1Regularization extends INDArrayWeights {
    def l1Regularization: INDArray
    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi { this: INDArrayOptimizer =>
      abstract override def delta: INDArray = super.delta + Transforms.sign(weight.data) * l1Regularization
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi with Optimizer
  }
  trait L2Regularization extends INDArrayWeights {
    def l2Regularization: INDArray
    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi { this: INDArrayOptimizer =>
      abstract override def delta: INDArray = super.delta + weight.data * l2Regularization
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi with Optimizer
  }

  trait Momentum extends INDArrayWeights {
    trait INDArrayWeightApi extends super.INDArrayWeightApi { this: INDArrayWeight =>
      def mu: scala.Double = 0.9
      var v: INDArray = Nd4j.zeros(data.shape: _*)
    }

    override type INDArrayWeight <: INDArrayWeightApi with Weight

    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi { this: INDArrayOptimizer =>

      private lazy val delta0: INDArray = {
        import weight._
        v = super.delta + v * mu
        v
      }
      abstract override def delta: INDArray = delta0
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi with Optimizer
  }

  trait NesterovMomentum extends Momentum {
    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi { this: INDArrayOptimizer =>
      abstract override lazy val delta: INDArray = {
        import weight._
        val vPrev = v
        vPrev * (-mu) + super.delta * (1 + mu)
      }
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi with Optimizer
  }

  /**
    * @note This [[Adagrad]] hyperparameter is usually used before global [[LearningRate]]. e.g. `Adagrad with FixedLearningRate`, not `FixedLearningRate with Adagrad`
    */
  trait Adagrad extends INDArrayWeights {
    trait INDArrayWeightApi extends super.INDArrayWeightApi { this: INDArrayWeight =>
      var cache: INDArray = Nd4j.zeros(data.shape: _*)
    }

    override type INDArrayWeight <: INDArrayWeightApi with Weight

    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi { this: INDArrayOptimizer =>
      def eps: scala.Double = 1e-4

      abstract override lazy val delta: INDArray = {
        import weight._
        cache = cache + super.delta * super.delta
        super.delta / (Transforms.sqrt(cache) + eps)
      }
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi with Optimizer
  }

  /**
    * @note This [[RMSprop]] hyperparameter is usually used before global [[LearningRate]]. e.g. `RMSprop with FixedLearningRate`, not `FixedLearningRate with RMSprop`
    */
  trait RMSprop extends INDArrayWeights {
    trait INDArrayWeightApi extends super.INDArrayWeightApi { this: INDArrayWeight =>
      var cache: INDArray = Nd4j.zeros(data.shape: _*)
    }

    override type INDArrayWeight <: INDArrayWeightApi with Weight

    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi { this: INDArrayOptimizer =>
      def eps: scala.Double = 1e-4
      def decayRate: scala.Double = 0.99
      abstract override lazy val delta: INDArray = {
        import weight._
        cache = cache * decayRate + super.delta * super.delta * (1.0 - decayRate)
        super.delta / (Transforms.sqrt(cache) + eps)
      }
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi with Optimizer
  }

  trait Adam extends INDArrayWeights {

    trait INDArrayWeightApi extends super.INDArrayWeightApi { this: INDArrayWeight =>
      var m: INDArray = Nd4j.zeros(data.shape: _*)
      var v: INDArray = Nd4j.zeros(data.shape: _*)
    }

    override type INDArrayWeight <: INDArrayWeightApi with Weight

    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi { this: INDArrayOptimizer =>
      def beta1: scala.Double = 0.9
      def beta2: scala.Double = 0.999
      def eps: scala.Double = 1e-8
      abstract override lazy val delta: INDArray = {
        import weight._
        m = m * beta1 + super.delta * (1.0 - beta1)
        v = v * beta2 + (super.delta * super.delta) * (1.0 - beta2)
        m / (Transforms.sqrt(v) + eps)
      }
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi with Optimizer
  }
}

/**
  * @author 杨博 (Yang Bo)
  */
class CumulativeINDArrayLayersSpec
    extends AsyncFreeSpec
    with Matchers
    with Inside
    with ThoughtworksFutureToScalaFuture {

  import CumulativeINDArrayLayersSpec._

  val hyperparameters = Factory[
    Logging with ImplicitsSingleton with DoubleTraining with INDArrayTraining with INDArrayLiterals with DoubleLiterals with CumulativeDoubleLayers with Operators with CumulativeINDArrayLayers with FixedLearningRate]
    .newInstance(fixedLearningRate = 1.0)

  import hyperparameters.implicits._

  def trainAndAssertLossAndWeight(myNetwork: INDArray => hyperparameters.INDArrayLayer,
                                  weight: hyperparameters.INDArrayWeight,
                                  trainTimes: Int = 2,
                                  expectedLoss: Int = 0,
                                  expectedWeightSum: Int = -16,
                                  input: INDArray = Nd4j.ones(4, 4)): Future[Assertion] = {
    @throwableMonadic[Future]
    val run: Future[Assertion] = {
      for (_ <- 1 to trainTimes) {
        myNetwork(input).train.each.sumT
      }
      val loss = myNetwork(input).predict.each
      loss.sumT should be(expectedLoss)
      weight.data.sumT should be(expectedWeightSum)
    }
    run
  }

  "INDArray + INDArray" in {

    val weight = hyperparameters.INDArrayWeight(Nd4j.ones(4, 4))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      weight + input
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "INDArray + Double" in {

    val weight = hyperparameters.INDArrayWeight(Nd4j.ones(4, 4))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      weight + 1.0
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "Double + INDArray" in {

    val weight = hyperparameters.INDArrayWeight(Nd4j.ones(4, 4))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      1.0 + weight
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "INDArray - INDArray" in {

    val weight = hyperparameters.INDArrayWeight(Nd4j.ones(4, 4))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      weight - (-input)
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "INDArray - Double" in {

    val weight = hyperparameters.INDArrayWeight(Nd4j.ones(4, 4))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      weight - (-1.0)
    }

    trainAndAssertLossAndWeight(myNetwork, weight)
  }

  "Double - INDArray" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4).negi()))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      1.0 - weight
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 16)
  }

  "INDArray * INDArray" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) mul 2))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      weight * input
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "INDArray * Double" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) mul 2))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      weight * 1.0
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "Double * INDArray" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) mul 2))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      1.0 * weight
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "INDArray / INDArray" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) mul 2))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      weight / input
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "INDArray / Double" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) mul 2))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      weight / 1.0
    }

    trainAndAssertLossAndWeight(myNetwork, weight, expectedWeightSum = 0)
  }

  "Double / INDArray" in {

    val weight = hyperparameters.INDArrayWeight(Nd4j.ones(4, 4))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      1.0 / weight
    }
    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 20000) {
        myNetwork(Nd4j.ones(4, 4)).train.each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork(Nd4j.ones(4, 4)).predict.each
      loss.sumT should be < 0.5
      weight.data.sumT should be > 600.0
    }

  }

  "max(INDArray,Double)" in {
    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      hyperparameters.max(weight, 0.0)
    }

    trainAndAssertLossAndWeight(myNetwork, weight, trainTimes = 10, expectedWeightSum = 0)

  }

  "min(INDArray,Double)" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      hyperparameters.min(weight, 100.0)
    }

    trainAndAssertLossAndWeight(myNetwork, weight, trainTimes = 10, expectedWeightSum = 0)
  }

  "exp(INDArray)" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      hyperparameters.exp(weight)
    }
    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 50) {
        myNetwork(Nd4j.ones(4, 4)).train.each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork(Nd4j.ones(4, 4)).predict.each
      loss.sumT should be < 1.0
      weight.data.sumT should be < 1.0
    }

  }

  "log(INDArray)" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      hyperparameters.log(weight)
    }
    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 50) {
        myNetwork(Nd4j.ones(4, 4)).train.each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork(Nd4j.ones(4, 4)).predict.each
      loss.sumT should be < 10.0
      weight.data.sumT should be < 22.0
    }

  }

  "abs(INDArray)" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      hyperparameters.abs(weight)
    }
    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 10) {
        myNetwork(Nd4j.ones(4, 4)).train.each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork(Nd4j.ones(4, 4)).predict.each
      loss.sumT should be < 1.0
      weight.data.sumT should be < 1.0
    }

  }

  "INDArray dot INDArray" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      def RichINDArray = ??? // Disable org.nd4s.Implicits.RichINDArray
      input dot weight
    }
    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 10) {
        myNetwork(Nd4j.ones(4, 4)).train.each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork(Nd4j.ones(4, 4)).predict.each
      loss.sumT should be(-1920.0)
      weight.data.sumT should be(-480.0)
    }

  }

  "INDArray im2col (kernel,stride,padding) --forward" in {
    val hyperparameters =
      Factory[
        CNNs with Logging with ImplicitsSingleton with INDArrayTraining with INDArrayLiterals with DoubleLiterals with CumulativeDoubleLayers with Operators with CumulativeINDArrayLayers with FixedLearningRate]
        .newInstance(fixedLearningRate = 0.03)
    import hyperparameters.implicits._
    val weight = hyperparameters.INDArrayWeight((-(1 to 54).toNDArray).reshape(2, 3, 3, 3))

    def myNetwork(kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int)): hyperparameters.INDArrayLayer = {
      hyperparameters.im2col(weight, kernel, stride, padding)
    }

    myNetwork((3, 3), (1, 1), (1, 1)).train.map { result =>
      result.sumT should be(-8085.0)
    }
  }

  "INDArray im2col (kernel,stride,padding) --train" in {
    val hyperparameters =
      Factory[
        CNNs with Logging with ImplicitsSingleton with INDArrayTraining with INDArrayLiterals with DoubleLiterals with CumulativeDoubleLayers with Operators with CumulativeINDArrayLayers with FixedLearningRate]
        .newInstance(fixedLearningRate = 0.01)
    import hyperparameters.implicits._
    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int)): hyperparameters.INDArrayLayer = {
      hyperparameters.im2col(weight, kernel, stride, padding)
    }
    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 1000) {
        myNetwork((3, 3), (1, 1), (1, 1)).train.each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = (myNetwork((3, 3), (1, 1), (1, 1))).predict.each
      loss.sumT should be < 1.0
    }
  }

  "INDArray reshape shapes --forward" in {

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(dimensions: Int*): hyperparameters.INDArrayLayer = {
      weight.reshape(dimensions: _*)
    }

    myNetwork(2, 3, 3, 3).train.map { result =>
      result.sumT should be(1431.0)
    }

  }

  "INDArray maxPool poolsize --forward" in {

    val hyperparameters =
      Factory[
        CNNs with Logging with ImplicitsSingleton with INDArrayTraining with INDArrayLiterals with DoubleLiterals with CumulativeDoubleLayers with Operators with CumulativeINDArrayLayers with FixedLearningRate]
        .newInstance(fixedLearningRate = 1.0)
    import hyperparameters.implicits._

    val weight = hyperparameters.INDArrayWeight((1 to 96).toNDArray.reshape(2, 3, 4, 4))

    def myNetwork(poolSize: (Int, Int)): hyperparameters.INDArrayLayer = {
      hyperparameters.maxPool(weight, poolSize)
    }

    myNetwork((2, 2)).train.map { result =>
      result.sumT should be(1224.0)
    }
  }

  "INDArray maxPool poolsize -- train" in {

    val hyperparameters =
      Factory[
        CNNs with Logging with ImplicitsSingleton with INDArrayTraining with INDArrayLiterals with DoubleLiterals with CumulativeDoubleLayers with Operators with CumulativeINDArrayLayers with FixedLearningRate]
        .newInstance(fixedLearningRate = 1.0)
    import hyperparameters.implicits._

    val weight = hyperparameters.INDArrayWeight((1 to 96).toNDArray.reshape(2, 3, 4, 4))

    def myNetwork(poolSize: (Int, Int)): hyperparameters.INDArrayLayer = {
      hyperparameters.maxPool(weight, poolSize)
    }

    val poolSize = (2, 2)

    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 700) {
        myNetwork(poolSize).train.each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss: INDArray = (myNetwork(poolSize)).predict.each
      loss.meanT should be < 10.0
    }

  }

  "4D INDArray * 4D INDArray -- forward" in {

    val weight = hyperparameters.INDArrayWeight((0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      weight * input
    }

    val input = (0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4)

    myNetwork(input).train.map { (result) =>
      result.meanNumber.doubleValue should be(180.16666666666666667 +- 0.1)
    }

  }

  "4D INDArray * 4D INDArray -- train" in {

    val weight = hyperparameters.INDArrayWeight((0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      weight * input
    }

    val input = (0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4)

    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 100) {
        myNetwork(input).train.each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss: INDArray = (myNetwork(input)).predict.each
      loss.meanNumber.doubleValue should be < 1.0
    }

  }

  "INDArray permute dimensions --forward" in {

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 9))

    def myNetwork(dimensions: Int*): hyperparameters.INDArrayLayer = {
      weight.permute(dimensions: _*)
    }

    myNetwork(0, 2, 1).train.map { (result) =>
      result.sumT should be(1431.0)
    }
  }

  "conv2d(INDArray, INDArray, INDArray, kernel, stride, padding)" in {

    val hyperparameters =
      Factory[
        CNNs with Logging with ImplicitsSingleton with INDArrayTraining with INDArrayLiterals with DoubleLiterals with CumulativeDoubleLayers with Operators with CumulativeINDArrayLayers with FixedLearningRate]
        .newInstance(fixedLearningRate = 0.01)
    import hyperparameters.implicits._

    val weight: hyperparameters.INDArrayWeight = hyperparameters.INDArrayWeight(Nd4j.ones(1, 1, 3, 3))
    val bias: hyperparameters.INDArrayWeight = hyperparameters.INDArrayWeight(Nd4j.zeros(1))

    def convolution(input: INDArray): hyperparameters.INDArrayLayer = {
      hyperparameters.conv2d(input, weight, bias, (3, 3), (1, 1), (1, 1))
    }

    val expectResult = Array(14.00, 24.00, 30.00, 22.00, 33.00, 54.00, 63.00, 45.00, 57.00, 90.00, 99.00, 69.00, 46.00,
      72.00, 78.00, 54.00).toNDArray
      .reshape(1, 1, 4, 4)

    val input: INDArray = (1 to 16).toNDArray.reshape(1, 1, 4, 4)
    convolution(input).train.map { (result) =>
      result.eq(expectResult).sumT should be(16)
    }

  }

  "sumT(INDArray)" in {

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(): hyperparameters.DoubleLayer = {
      weight.sum
    }
    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 54) {
        myNetwork().train.each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork().predict.each
      loss should be < 1.0
    }

  }

  "sum(INDArray,dimensions) --2 dimensions" in {

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(6, 9))

    def myNetwork(dimensions: Int*): hyperparameters.INDArrayLayer = {
      weight.sum(dimensions: _*)
    }

    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 54) {
        myNetwork(0).train.each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork(0).predict.each
      loss.sumT should be < 1.0
    }

  }

  // Failed due to nd4j bugs in broadcasting. TODO: Try to upgrade nd4j to a new version.
  "sum(INDArray,dimensions) --4 dimensions" in {

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(dimensions: Int*): hyperparameters.INDArrayLayer = {
      weight.sum(dimensions: _*)
    }
    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 54) {
        myNetwork(0, 1).train.each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork(0, 1).predict.each
      loss.sumT should be < 1.0
    }

  }

  "mean(INDArray)" in {

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(): hyperparameters.DoubleLayer = {
      weight.mean
    }

    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 27) {
        myNetwork().train.each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork().predict.each
      loss should be < 27.5
    }

  }

  "INDArray reshape shapes --train" in {
    val hyperparameters =
      Factory[Logging with ImplicitsSingleton with INDArrayTraining with INDArrayLiterals with DoubleLiterals with CumulativeDoubleLayers with Operators with CumulativeINDArrayLayers with FixedLearningRate]
        .newInstance(fixedLearningRate = 0.01)
    import hyperparameters.implicits._
    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(dimensions: Int*): hyperparameters.INDArrayLayer = {
      weight.reshape(dimensions: _*)
    }

    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 10000) {
        myNetwork(2, 3, 3, 3).train.each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = (myNetwork(2, 3, 3, 3)).predict.each
      loss.sumT should be < 1.0
    }

  }

  "INDArray permute dimensions --train" in {
    val hyperparameters =
      Factory[Logging with ImplicitsSingleton with INDArrayTraining with INDArrayLiterals with DoubleLiterals with CumulativeDoubleLayers with Operators with CumulativeINDArrayLayers with FixedLearningRate]
        .newInstance(fixedLearningRate = 0.01)
    import hyperparameters.implicits._
    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 9))

    def myNetwork(dimensions: Int*): hyperparameters.INDArrayLayer = {
      weight.permute(dimensions: _*)
    }

    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 10000) {
        myNetwork(0, 2, 1).train.each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork(0, 2, 1).predict.each
      loss.sumT should be < 1.0
    }

  }
}
