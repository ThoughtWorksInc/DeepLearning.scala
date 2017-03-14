package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.DifferentiableInt._
import com.thoughtworks.deeplearning.Layer.Tape
import com.thoughtworks.deeplearning.Symbolic.Layers.{Identity, Literal}
import org.scalatest.{FreeSpec, Matchers}
import com.thoughtworks.deeplearning.Poly.MathOps
import com.thoughtworks.deeplearning.Poly.MathFunctions._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DifferentiableIntSpec extends FreeSpec with Matchers {

  "Given a network: w + 1 (the initial value of w is 5)" - {
    implicit val optimizer = new DifferentiableInt.Optimizers.LearningRate {
      override protected def currentLearningRate() = 0.6
    }

    def makeNetwork(implicit x: Placeholder[Any, ExistentialNothing]) = {
      val w = 5.toWeight
      w + 1
    }

    val neuralNetwork = makeNetwork
    val DifferentiableInt.Layers.Plus(DifferentiableInt.Layers.Weight(initialW), Literal(1)) = neuralNetwork

    "w should be 5 at beginning" in {
      initialW should be(5)
    }

    "When training the network for many iterations" - {

      for (i <- 0 until 100) {
        neuralNetwork.train(())
      }

      "w should be -1" in {
        neuralNetwork should be(DifferentiableInt.Layers.Plus(DifferentiableInt.Layers.Weight(-1), Literal(1)))
      }

      "the result should be 0" in {
        neuralNetwork.predict(()) should be(0)
      }

    }

  }

}
