package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.BpAny._
import com.thoughtworks.deeplearning.BpDouble._
import com.thoughtworks.deeplearning.Layer.Batch
import org.scalatest.{FreeSpec, Matchers}
import com.thoughtworks.deeplearning.Poly.MathOps
import com.thoughtworks.deeplearning.Poly.MathFunctions._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class OperatorSpec extends FreeSpec with Matchers {

  import OperatorSpec._

  "DoubleBackProgationType input" in {

    def buildLayer(implicit input: shapeless.the.`Parameter[Double]`.Out): D2D.Out = {
      val m0: D2D.Out = 0.0 - max(1.0, 2.0) - input
      -m0
    }

    toAnyLayerOps(buildLayer).train(1.0)
  }

}

object OperatorSpec {
  val D2D = shapeless.the[Double <=> Double]
}
