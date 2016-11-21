package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.any._
import com.thoughtworks.deeplearning.double._
import org.scalatest.{FreeSpec, Matchers}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class OperatorSpec extends FreeSpec with Matchers {

  "double input" in {
    def buildLayer(implicit input: Double) = {
      val m0 = 0.0 - max(1.0, 2.0)
    }

    buildLayer
  }

}
