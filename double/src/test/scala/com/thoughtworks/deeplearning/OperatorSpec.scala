package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.dsl._
import com.thoughtworks.deeplearning.double._
import org.scalatest.{FreeSpec, Matchers}
import com.thoughtworks.deeplearning.Poly.LayerPolyOps
import com.thoughtworks.deeplearning.Poly.PolyFunctions._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class OperatorSpec extends FreeSpec with Matchers {

  "double input" in {
    def buildLayer(implicit input: BpDouble) = {
      val m0 = 0.0 - max(1.0, 2.0)
    }

    buildLayer
  }

}
