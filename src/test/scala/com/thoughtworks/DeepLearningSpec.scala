package com.thoughtworks

import org.scalatest._
import cats.implicits._
import com.thoughtworks.DeepLearning.DifferentiableFunction.Id
import shapeless.DepFn1

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DeepLearningSpec extends FreeSpec with Matchers with Inside {

  "-" in {
    implicit def learningRate = new DeepLearning.LearningRate {
      def apply() = 0.03
    }

    def f(dsl: Dsl)(input: dsl.Double): dsl.Double = {
      import dsl._
      -input - Double.weight(3.0)
    }
    def g(dsl: Dsl)(input: dsl.Double): dsl.Double = {
      import dsl._
      input - Double.weight(1.2)
    }

    val id = new Id[scala.Double, scala.Double]
    val dsl = new DeepLearning[id.Input]
    val f1 = f(dsl)(id)
    val g1 = g(dsl)(id)
    val nn = f1.compose(g1)
    def train(input: scala.Double): scala.Double = {
      val output = nn.forward(dsl.Double(10.11))
      output.backward(output.value)
      output.value
    }
    val initialLoss = train(10.11)
    for (_ <- 0 until 100) {
      train(10.11)
    }
    math.abs(train(10.11)) should be < math.abs(initialLoss)

    math.abs(train(10.11)) should be < 0.5
  }

}
