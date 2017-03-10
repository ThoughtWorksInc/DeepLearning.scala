package com.thoughtworks
package deeplearning

import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.DifferentiableHList._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.DifferentiableAny._
import org.scalatest.{FreeSpec, Matchers}
import ToLayer._
import com.thoughtworks.deeplearning.Poly.MathOps
import com.thoughtworks.deeplearning.Poly.MathFunctions._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
@enableMembersIf(!scala.util.Properties.versionNumberString.startsWith("2.12."))
final class MaxSpec extends FreeSpec with Matchers {

  import com.thoughtworks.deeplearning.DifferentiableINDArray._

  "complex input" in {
    def buildNeuralNetwork(implicit input: DoublePlaceholder :**: INDArrayPlaceholder :**: HNilPlaceholder) = {
      val m0 = max(1.0, 2.0.toLayer)
      val m1: input.To[DoublePlaceholder] = max(m0, 1.6)
      val m2 = max(m0.toLayer, m1.toLayer)
    }

    buildNeuralNetwork
  }

  "DoublePlaceholder input" in {
    def buildNeuralNetwork(implicit input: DoublePlaceholder) = {
      val m0 = max(1.0, 2.0)
      val m1: input.To[DoublePlaceholder] = max(input, 1.6)
      val m2 = max(m0, m1)
      val m3 = max(0.0, max(m0, max(max(input, m1), m2)))
      val m4: input.To[DoublePlaceholder] = max(1.6, input)
    }

    buildNeuralNetwork
  }

  "INDArrayPlaceholder input" in {
    def buildNeuralNetwork(implicit input: INDArrayPlaceholder) = {
      val m0: input.To[DoublePlaceholder] = max(1.0, 2.0)
      val m1: input.To[INDArrayPlaceholder] = max(input, 1.6)
      val m2 = max(input, m0)
      val m3: input.To[DoublePlaceholder] = max(m0, 2.0)
      val m4 = max(max(max(max(m1, 3.0), m3), m0), 7.2)

      "max(0.1, input)" shouldNot compile
    }

    buildNeuralNetwork
  }

}
