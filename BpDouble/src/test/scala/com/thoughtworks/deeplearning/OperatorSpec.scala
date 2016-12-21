package com.thoughtworks.deeplearning

import cats.Eval
import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.Poly._
import com.thoughtworks.deeplearning.BpDouble._
import com.thoughtworks.deeplearning.Layer.Batch
import org.scalatest.{FreeSpec, Matchers}
import com.thoughtworks.deeplearning.Poly.MathOps
import com.thoughtworks.deeplearning.Poly.MathFunctions._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class OperatorSpec extends FreeSpec with Matchers {

  "BpDouble input" in {

    def buildLayer(
        implicit input: shapeless.the.`Parameter[Double]`.Out): shapeless.the.`Double LayerOf Double`.Out = {
      val m0 = 0.0 - max(1.0, 2.0)
      -m0 * m0
    }

    buildLayer
  }

}
