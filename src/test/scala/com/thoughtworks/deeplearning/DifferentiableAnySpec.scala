package com.thoughtworks.deeplearning

import cats._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.Poly.MathOps
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import org.scalatest.{FreeSpec, Matchers}
import shapeless._

/**
  * Created by 张志豪 on 2017/1/23.
  */
class DifferentiableAnySpec extends FreeSpec with Matchers {
  "withOutputDataHook" in {
    def layer1(implicit x: From[Double]##`@`) = {
      x + x
    }

    var count = 0
    val layer2 = layer1.withOutputDataHook { x: Double =>
      x should be(2.4)
      count += 1
    }
    layer2.train(1.2)
    count should be(1)
  }
}
