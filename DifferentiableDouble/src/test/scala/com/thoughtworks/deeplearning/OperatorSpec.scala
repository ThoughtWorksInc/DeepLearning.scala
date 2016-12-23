package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.Layer.Batch
import org.scalatest.{FreeSpec, Matchers}
import com.thoughtworks.deeplearning.Poly.MathOps
import com.thoughtworks.deeplearning.Poly.MathFunctions._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class OperatorSpec extends FreeSpec with Matchers {

  "DoublePlaceholder input" in {
//    shapeless.the.`From[Double]`.Out
    def buildLayer(implicit input: shapeless.the.`From[Double]`.Out): shapeless.the.`To[Double]`.Out = {
      val m0: shapeless.the.`To[Double]`.Out = 0.0 - max(1.0, 2.0) - input
      -m0
    }

    val doubleToDouble = FromTo[Double, Double]
    val layer: shapeless.the.`Double <=> Double`.Out = buildLayer

    (layer: doubleToDouble.Out).train(1.0)
  }

}
