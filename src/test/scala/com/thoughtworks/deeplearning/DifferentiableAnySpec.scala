package com.thoughtworks.deeplearning

import cats._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.Poly.MathOps
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import org.scalatest.{FreeSpec, Matchers}
import shapeless._

/**
  * Created by 张志豪 on 2017/1/23.
  */
class DifferentiableAnySpec extends FreeSpec with Matchers {
  "withOutputDataHook" in {
    def makeNetwork1(implicit x: From[Double]##T) = {
      x + x
    }

    makeNetwork1.withOutputDataHook { x: Double => println(x) }
  }
}
