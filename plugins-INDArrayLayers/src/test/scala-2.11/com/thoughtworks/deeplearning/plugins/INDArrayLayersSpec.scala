package com.thoughtworks.deeplearning
package plugins

import com.thoughtworks.each.Monadic._
import com.thoughtworks.feature.{Factory, ImplicitApply}
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.scalatest._
import org.nd4s.Implicits._

import scala.concurrent.Promise
import scalaz.{-\/, \/, \/-}
import scalaz.concurrent.Task
import scalaz.std.iterable._

object INDArrayLayersSpec {

  trait CNNs extends RawINDArrayLayers with ImplicitsSingleton with Training with Operators {

    trait ImplicitsApi
        extends super[RawINDArrayLayers].ImplicitsApi
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
class INDArrayLayersSpec extends AsyncFreeSpec with Matchers with Inside {

  import INDArrayLayersSpec._

  val hyperparameters = Factory[
    Logging with ImplicitsSingleton with DoubleTraining with INDArrayTraining with INDArrayLiterals with DoubleLiterals with DoubleLayers with Operators with INDArrayLayers with FixedLearningRate]
    .newInstance(fixedLearningRate = 1.0)

  import hyperparameters.implicits._

  def trainAndAssertLossAndWeight(myNetwork: INDArray => hyperparameters.INDArrayLayer,
                                  weight: hyperparameters.INDArrayWeight,
                                  trainTimes: Int = 2,
                                  expectedLoss: Int = 0,
                                  expectedWeightSum: Int = -16,
                                  input: INDArray = Nd4j.ones(4, 4)): scala.concurrent.Future[Assertion] = {
    @throwableMonadic[Task]
    val run: Task[Assertion] = {
      for (_ <- 1 to trainTimes) {
        myNetwork(input).train.each.sumT
      }
      val loss = myNetwork(input).predict.each
      loss.sumT should be(expectedLoss)
      weight.data.sumT should be(expectedWeightSum)
    }
    val p = Promise[Assertion]

    run.unsafePerformAsync {
      case -\/(e) =>
        p.failure(e)
      case \/-(assersion) =>
        p.success(assersion)
    }
    p.future
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

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 20000) {
        myNetwork(Nd4j.ones(4, 4)).train.each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      myNetwork(Nd4j.ones(4, 4)).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be < 0.5
            weight.data.sumT should be > 600.0
        }
      }
    }
    p.future
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

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 50) {
        myNetwork(Nd4j.ones(4, 4)).train.each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      myNetwork(Nd4j.ones(4, 4)).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be < 1.0
            weight.data.sumT should be < 1.0
        }
      }
    }
    p.future
  }

  "log(INDArray)" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      hyperparameters.log(weight)
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 50) {
        myNetwork(Nd4j.ones(4, 4)).train.each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      myNetwork(Nd4j.ones(4, 4)).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be < 10.0
            weight.data.sumT should be < 22.0
        }
      }
    }
    p.future
  }

  "abs(INDArray)" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      hyperparameters.abs(weight)
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 10) {
        myNetwork(Nd4j.ones(4, 4)).train.each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      myNetwork(Nd4j.ones(4, 4)).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be < 1.0
            weight.data.sumT should be < 1.0
        }
      }
    }
    p.future
  }

  "INDArray dot INDArray" in {

    val weight = hyperparameters.INDArrayWeight((Nd4j.ones(4, 4) * 10))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      def RichINDArray = ??? // Disable org.nd4s.Implicits.RichINDArray
      input dot weight
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 10) {
        myNetwork(Nd4j.ones(4, 4)).train.each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      myNetwork(Nd4j.ones(4, 4)).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be(-1920.0)
            weight.data.sumT should be(-480.0)
        }
      }
    }
    p.future
  }

  "INDArray im2col (kernel,stride,padding) --forward" in {
    val hyperparameters =
      Factory[
        CNNs with Logging with ImplicitsSingleton with INDArrayTraining with INDArrayLiterals with DoubleLiterals with DoubleLayers with Operators with INDArrayLayers with FixedLearningRate]
        .newInstance(fixedLearningRate = 0.03)
    import hyperparameters.implicits._
    val weight = hyperparameters.INDArrayWeight((-(1 to 54).toNDArray).reshape(2, 3, 3, 3))

    def myNetwork(kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int)): hyperparameters.INDArrayLayer = {
      hyperparameters.im2col(weight, kernel, stride, padding)
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val p = Promise[Assertion]

    myNetwork((3, 3), (1, 1), (1, 1)).train.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(result) =>
            result.sumT should be(-8085.0)
        }
      }
    }

    p.future
  }

  "INDArray im2col (kernel,stride,padding) --train" in {
    val hyperparameters =
      Factory[
        CNNs with Logging with ImplicitsSingleton with INDArrayTraining with INDArrayLiterals with DoubleLiterals with DoubleLayers with Operators with INDArrayLayers with FixedLearningRate]
        .newInstance(fixedLearningRate = 0.01)
    import hyperparameters.implicits._
    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int)): hyperparameters.INDArrayLayer = {
      hyperparameters.im2col(weight, kernel, stride, padding)
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 1000) {
        myNetwork((3, 3), (1, 1), (1, 1)).train.each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      (myNetwork((3, 3), (1, 1), (1, 1))).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be < 1.0
        }
      }
    }
    p.future
  }

  "INDArray reshape shapes --forward" in {

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(dimensions: Int*): hyperparameters.INDArrayLayer = {
      weight.reshape(dimensions: _*)
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val p = Promise[Assertion]

    myNetwork(2, 3, 3, 3).train.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(result) =>
            result.sumT should be(1431.0)
        }
      }
    }

    p.future
  }
  "4D INDArray * 4D INDArray -- forward" in {

    val weight = hyperparameters.INDArrayWeight((0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      weight * input
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val input = (0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4)

    val p = Promise[Assertion]

    myNetwork(input).train.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(result) =>
            result.meanNumber.doubleValue should be(180.16666666666666667 +- 0.1)
        }
      }
    }
    p.future
  }

  "4D INDArray * 4D INDArray -- train" in {

    val weight = hyperparameters.INDArrayWeight((0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4))

    def myNetwork(input: INDArray): hyperparameters.INDArrayLayer = {
      weight * input
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val input = (0 until (1 * 2 * 3 * 4)).toNDArray.reshape(1, 2, 3, 4)

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 100) {
        myNetwork(input).train.each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      (myNetwork(input)).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss: INDArray) =>
            loss.meanNumber.doubleValue should be < 1.0
        }
      }
    }
    p.future
  }

  "INDArray permute dimensions --forward" in {

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 9))

    def myNetwork(dimensions: Int*): hyperparameters.INDArrayLayer = {
      weight.permute(dimensions: _*)
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val p = Promise[Assertion]

    myNetwork(0, 2, 1).train.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(result) =>
            result.sumT should be(1431.0)
        }
      }
    }

    p.future
  }

  "conv2d(INDArray, INDArray, INDArray, kernel, stride, padding)" in {

    val hyperparameters =
      Factory[
        CNNs with Logging with ImplicitsSingleton with INDArrayTraining with INDArrayLiterals with DoubleLiterals with DoubleLayers with Operators with INDArrayLayers with FixedLearningRate]
        .newInstance(fixedLearningRate = 0.01)
    import hyperparameters.implicits._

    val weight: hyperparameters.INDArrayWeight = hyperparameters.INDArrayWeight(Nd4j.ones(1, 1, 3, 3))
    val bias: hyperparameters.INDArrayWeight = hyperparameters.INDArrayWeight(Nd4j.zeros(1))

    def convolution(input: INDArray): hyperparameters.INDArrayLayer = {
      hyperparameters.conv2d(input, weight, bias, (3, 3), (1, 1), (1, 1))
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    val p = Promise[Assertion]

    val expectResult = Array(14.00, 24.00, 30.00, 22.00, 33.00, 54.00, 63.00, 45.00, 57.00, 90.00, 99.00, 69.00, 46.00,
      72.00, 78.00, 54.00).toNDArray
      .reshape(1, 1, 4, 4)

    val input: INDArray = (1 to 16).toNDArray.reshape(1, 1, 4, 4)
    val task: Task[INDArray] = convolution(input).train
    task.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(result) =>
            result.eq(expectResult).sumT should be(16)
        }
      }
    }

    p.future
  }

  "sumT(INDArray)" in {

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(): hyperparameters.DoubleLayer = {
      weight.sum
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 54) {
        myNetwork().train.each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      myNetwork().predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Double] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss should be < 1.0
        }
      }
    }
    p.future
  }

  "sum(INDArray,dimensions) --2 dimensions" in {

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(6, 9))

    def myNetwork(dimensions: Int*): hyperparameters.INDArrayLayer = {
      weight.sum(dimensions: _*)
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 54) {
        myNetwork(0).train.each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      myNetwork(0).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be < 1.0
        }
      }
    }
    p.future
  }

  // Failed due to nd4j bugs in broadcasting. TODO: Try to upgrade nd4j to a new version.
  "sum(INDArray,dimensions) --4 dimensions" in {

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(dimensions: Int*): hyperparameters.INDArrayLayer = {
      weight.sum(dimensions: _*)
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 54) {
        myNetwork(0, 1).train.each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      (myNetwork(0, 1)).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be < 1.0
        }
      }
    }
    p.future
  }

  "mean(INDArray)" in {

    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(): hyperparameters.DoubleLayer = {
      weight.mean
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 27) {
        myNetwork().train.each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      (myNetwork()).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, Double] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss should be < 27.5
        }
      }
    }
    p.future
  }

  "INDArray reshape shapes --train" in {
    val hyperparameters =
      Factory[Logging with ImplicitsSingleton with INDArrayTraining with INDArrayLiterals with DoubleLiterals with DoubleLayers with Operators with INDArrayLayers with FixedLearningRate]
        .newInstance(fixedLearningRate = 0.01)
    import hyperparameters.implicits._
    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))

    def myNetwork(dimensions: Int*): hyperparameters.INDArrayLayer = {
      weight.reshape(dimensions: _*)
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 10000) {
        myNetwork(2, 3, 3, 3).train.each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      (myNetwork(2, 3, 3, 3)).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be < 1.0
        }
      }
    }
    p.future
  }

  "INDArray permute dimensions --train" in {
    val hyperparameters =
      Factory[Logging with ImplicitsSingleton with INDArrayTraining with INDArrayLiterals with DoubleLiterals with DoubleLayers with Operators with INDArrayLayers with FixedLearningRate]
        .newInstance(fixedLearningRate = 0.01)
    import hyperparameters.implicits._
    val weight = hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 9))

    def myNetwork(dimensions: Int*): hyperparameters.INDArrayLayer = {
      weight.permute(dimensions: _*)
    }

    import scalaz.concurrent.Future._
    import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

    @monadic[Task]
    val task: Task[Unit] = {
      for (_ <- 1 to 10000) {
        myNetwork(0, 2, 1).train.each
      }
    }

    val result = throwableMonadic[Task] {
      task.each
      (myNetwork(0, 2, 1)).predict.each
    }

    val p = Promise[Assertion]

    result.unsafePerformAsync { either: \/[Throwable, INDArray] =>
      p.success {
        inside(either) {
          case -\/(e) => throw e
          case \/-(loss) =>
            loss.sumT should be < 1.0
        }
      }
    }
    p.future
  }
}
