package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.Layer.Tape
import org.scalatest.{FreeSpec, Matchers}
import com.thoughtworks.deeplearning.Poly.MathOps
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import shapeless._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class OperatorSpec extends FreeSpec with Matchers {

  "DoublePlaceholder input" in {
    def buildLayer(implicit input: Double @Symbolic): Double @Symbolic = {
      val m0: To[Double]##`@` = 0.0 - max(1.0, 2.0) - input
      val layer: Layer.Aux[Tape.Aux[Double, Double], Tape.Aux[Double, Double]] = -m0
      val layer2: (Double => Double) @Symbolic = layer

      val layer3: To[Double]##`@` = layer2

      val d = To[Double]
      val layer4: d.`@` = layer3

      layer4
    }

    val doubleToDouble = FromTo[Double, Double]
    val layer: (Double => Double) @Symbolic = buildLayer

    (layer: doubleToDouble.`@`).train(1.0)
  }

}
