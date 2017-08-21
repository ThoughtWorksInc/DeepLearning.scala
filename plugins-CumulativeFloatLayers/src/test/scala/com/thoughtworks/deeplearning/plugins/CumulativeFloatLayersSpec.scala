package com.thoughtworks.deeplearning
package plugins

import com.thoughtworks.feature.Factory
import com.thoughtworks.each.Monadic._
import org.scalactic.ErrorMessage
import org.scalatest._
import com.thoughtworks.future._
import com.thoughtworks.deeplearning.scalatest.ThoughtworksFutureToScalaFuture
import com.thoughtworks.feature.mixins.ImplicitsSingleton

import scalaz.std.iterable._

object CumulativeFloatLayersSpec {

  trait FixedLearningRate extends LearningRate {
    def fixedLearningRate: scala.Float
    trait FloatOptimizerApi extends super.FloatOptimizerApi { this: FloatOptimizer =>
      final def learningRate: scala.Float = fixedLearningRate
    }
    override type FloatOptimizer <: FloatOptimizerApi with Optimizer
  }

  trait LearningRate extends FloatWeights {
    trait FloatOptimizerApi extends super.FloatOptimizerApi { this: FloatOptimizer =>
      def learningRate: scala.Float
      override def delta: scala.Float = super.delta * learningRate
    }
    override type FloatOptimizer <: FloatOptimizerApi with Optimizer
  }

  trait L1Regularization extends FloatWeights {
    def l1Regularization: scala.Float
    trait FloatOptimizerApi extends super.FloatOptimizerApi { this: FloatOptimizer =>
      override def delta: scala.Float = super.delta + scala.math.signum(weight.data) * l1Regularization
    }
    override type FloatOptimizer <: FloatOptimizerApi with Optimizer
  }
  trait L2Regularization extends FloatWeights {
    def l2Regularization: scala.Float
    trait FloatOptimizerApi extends super.FloatOptimizerApi { this: FloatOptimizer =>
      override def delta: scala.Float = super.delta + weight.data * l2Regularization
    }
    override type FloatOptimizer <: FloatOptimizerApi with Optimizer
  }

  case class Boom(errorMessage: ErrorMessage) extends RuntimeException

}

/**
  * @author 杨博 (Yang Bo)
  */
final class CumulativeFloatLayersSpec
    extends AsyncFreeSpec
    with Matchers
    with Inside
    with ThoughtworksFutureToScalaFuture {
  // TODO: Add tests for exception handling

  import CumulativeFloatLayersSpec._

  "Plus" in {

    val hyperparameters =
      Factory[FloatTraining with Operators with FloatLiterals with CumulativeFloatLayers with ImplicitsSingleton with FixedLearningRate]
        .newInstance(fixedLearningRate = 1.0f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      6.7f + input + weight + 5.5f
    }

    def train(inputData: Float): Future[Float] = {
      myNetwork(inputData).train
    }

    throwableMonadic[Future] {
      train(1.0f).each
      train(1.0f).each
      train(1.0f).each
      train(1.0f).each
      train(1.0f).each
      weight.data should be(-4)
    }

  }

  "Plus with Predict" in {
    val hyperparameters =
      Factory[FloatTraining with Operators with FloatLiterals with CumulativeFloatLayers with ImplicitsSingleton with FixedLearningRate]
        .newInstance(fixedLearningRate = 1.0f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      1.0f + input + weight + 4.0f
    }

    def trainMyNetwork(inputData: Float): Future[Float] = {
      myNetwork(inputData).train
    }

    throwableMonadic[Future] {
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      trainMyNetwork(1.0f).each
      val loss = myNetwork(1.0f).predict.each
      loss should be(0.0f)
    }

  }

  "Predict -- use for" in {
    val hyperparameters =
      Factory[FloatTraining with Operators with FloatLiterals with CumulativeFloatLayers with ImplicitsSingleton with FixedLearningRate]
        .newInstance(fixedLearningRate = 1.0f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      -10.0f + 20.0f - ((input - weight + 4.0f) * 2.0f / 2.0f)
      //10.0f - (input - weight + 4.0f) //6
    }

    def trainMyNetwork(inputData: Float): Future[Float] = {
      myNetwork(inputData).train
    }

    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 6) {
        trainMyNetwork(1.0f).each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork(1.0f).predict.each
      loss should be(0.0f)
      weight.data should be(-5)
    }

  }

  "will not stackOverFlow" in {
    val hyperparameters =
      Factory[FloatTraining with Operators with FloatLiterals with CumulativeFloatLayers with ImplicitsSingleton with FixedLearningRate]
        .newInstance(fixedLearningRate = 1.0f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      -10.0f + 20.0f - ((input - weight + 4.0f) * 2.0f / 2.0f)
    }

    def trainMyNetwork(inputData: Float): Future[Float] = {
      myNetwork(inputData).train
    }

    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 1000) {
        Future.execute(()).each
        trainMyNetwork(1.0f).each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork(1.0f).predict.each
      true should be(true)
    }

  }

  "min" in {
    val hyperparameters =
      Factory[FloatTraining with Operators with FloatLiterals with CumulativeFloatLayers with ImplicitsSingleton with FixedLearningRate]
        .newInstance(fixedLearningRate = 1.0f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      5.0f - hyperparameters.min(5.0f, weight)
    }

    def trainMyNetwork(inputData: Float): Future[Float] = {
      myNetwork(inputData).train
    }

    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 4) {
        trainMyNetwork(1.0f).each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork(1.0f).predict.each
      loss should be(0.0f)
      weight.data should be(5)
    }

  }

  "max" in {
    val hyperparameters =
      Factory[FloatTraining with Operators with FloatLiterals with CumulativeFloatLayers with ImplicitsSingleton with FixedLearningRate]
        .newInstance(fixedLearningRate = 1.0f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      10.0f - hyperparameters.max(0.0f, weight)
    }

    def trainMyNetwork(inputData: Float): Future[Float] = {
      myNetwork(inputData).train
    }

    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 9) {
        trainMyNetwork(1.0f).each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork(1.0f).predict.each
      loss should be(0.0f)
      weight.data should be(10)
    }

  }

  "log" in {
    val hyperparameters =
      Factory[FloatTraining with Operators with FloatLiterals with CumulativeFloatLayers with ImplicitsSingleton with FixedLearningRate]
        .newInstance(fixedLearningRate = 0.5f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    val log5 = scala.math.log(5).toFloat

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      log5 - hyperparameters.log(weight)
    }

    def trainMyNetwork(inputData: Float): Future[Float] = {
      myNetwork(inputData).train
    }

    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 23) {
        Future.execute(()).each
        trainMyNetwork(1.0f).each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork(1.0f).predict.each
      scala.math.abs(weight.data - 5) should be < 0.1f
      loss should be < 0.1f
    }
  }

  "exp" in {
    val hyperparameters =
      Factory[FloatTraining with Operators with FloatLiterals with CumulativeFloatLayers with ImplicitsSingleton with FixedLearningRate]
        .newInstance(fixedLearningRate = 0.1f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    val exp3 = scala.math.exp(3).toFloat

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      exp3 - hyperparameters.exp(weight)
    }

    def trainMyNetwork(inputData: Float): Future[Float] = {
      myNetwork(inputData).train
    }

    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 4) {
        trainMyNetwork(1.0f).each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork(1.0f).predict.each
      scala.math.abs(weight.data - 3) should be < 0.1f
      loss should be < 0.5f
    }

  }

  "abs" in {
    val hyperparameters =
      Factory[FloatTraining with Operators with FloatLiterals with CumulativeFloatLayers with ImplicitsSingleton with FixedLearningRate]
        .newInstance(fixedLearningRate = 1.0f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      5.0f - hyperparameters.abs(weight)
    }

    def trainMyNetwork(inputData: Float): Future[Float] = {
      myNetwork(inputData).train
    }

    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 4) {
        trainMyNetwork(1.0f).each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork(1.0f).predict.each
      weight.data should be(5.0f)
      loss should be(0)
    }

  }

  "unary_-" in {
    val hyperparameters =
      Factory[FloatTraining with Operators with FloatLiterals with CumulativeFloatLayers with ImplicitsSingleton with FixedLearningRate]
        .newInstance(fixedLearningRate = 1.0f)
    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(5.0f)

    def myNetwork(input: Float): hyperparameters.FloatLayer = {
      hyperparameters.abs(-weight)
    }

    def trainMyNetwork(inputData: Float): Future[Float] = {
      myNetwork(inputData).train
    }

    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 5) {
        trainMyNetwork(1.0f).each
      }
    }

    throwableMonadic[Future] {
      task.each
      val loss = myNetwork(1.0f).predict.each
      weight.data should be(0.0f)
      loss should be(0)
    }

  }
}
