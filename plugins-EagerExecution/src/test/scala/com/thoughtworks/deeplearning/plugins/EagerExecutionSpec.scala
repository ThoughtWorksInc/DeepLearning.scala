package com.thoughtworks.deeplearning
package plugins

import com.thoughtworks.deeplearning.plugins.EagerExecution.Eager
import com.thoughtworks.feature.Factory
import org.scalactic.ErrorMessage
import org.scalatest._
import com.thoughtworks.future._
import com.thoughtworks.deeplearning.scalatest.ThoughtworksFutureToScalaFuture
import scalaz.std.iterable._

object EagerExecutionSpec {

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

}

/**
  * @author 杨博 (Yang Bo)
  */
final class EagerExecutionSpec extends AsyncFreeSpec with Matchers with Inside with ThoughtworksFutureToScalaFuture {

  import EagerExecutionSpec._

  "EagerExecution" in {

    val hyperparameters =
      Factory[FloatTraining with EagerExecution with Operators with FloatLiterals with CumulativeFloatLayers with ImplicitsSingleton with FixedLearningRate]
        .newInstance(fixedLearningRate = 1.0f)

    import hyperparameters.FloatLayer

    import hyperparameters.implicits._

    val weight = hyperparameters.FloatWeight(1.0f)

    def myNetwork(input: Float): FloatLayer = {
      // FIXME: inlining !-notation does not compile due to https://github.com/ThoughtWorksInc/Dsl.scala/issues/119
      // 6.7f + !Eager(input + weight) + weight + 5.5f

      val f = !Eager(input + weight)
      6.7f + f + weight + 5.5f
    }: @com.thoughtworks.dsl.Dsl.reset

    def train(inputData: Float): Future[Float] = {
      myNetwork(inputData).train
    }

    for {
      _ <- train(1.0f)
      _ <- train(1.0f)
      _ <- train(1.0f)
      _ <- train(1.0f)
      _ <- train(1.0f)
    } yield weight.data should be(-4)

  }

}
