package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.array2D._
import com.thoughtworks.deeplearning.dsl._
import com.thoughtworks.deeplearning.hlist._
import com.thoughtworks.deeplearning.double._
import org.scalatest.{FreeSpec, Matchers}
import ToLayer._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class MaxSpec extends FreeSpec with Matchers {

  "complex input" in {
    def buildNeuralNetwork(implicit input: BpDouble :**: Array2D :**: BpHNil) = {
      val m0 = max(1.0, 2.0.toLiteral)
      val m1: input.To[BpDouble] = max(m0, 1.6)
      val m2 = max(m0.toLiteral, m1.toLiteral)
    }

    buildNeuralNetwork
  }

  "double input" in {
    def buildNeuralNetwork(implicit input: BpDouble) = {
      val m0 = max(1.0, 2.0)
      val m1: input.To[BpDouble] = max(input, 1.6)
      val m2 = max(m0, m1)
      val m3 = max(0.0, max(m0, max(max(input, m1), m2)))
      val m4: input.To[BpDouble] = max(1.6, input)
    }

    buildNeuralNetwork
  }

  "array2D input" in {
    def buildNeuralNetwork(implicit input: Array2D) = {
      val m0 = max(1.0, 2.0)
      val m1: input.To[Array2D] = max(input, 1.6)
      val m2 = max(input, m0)
      val m3: input.To[BpDouble] = max(m0, 2.0)
      val m4 = max(max(max(max(m1, 3.0), m3), m0), 7.2)

      "max(0.1, input)" shouldNot compile
    }

    buildNeuralNetwork
  }

}
