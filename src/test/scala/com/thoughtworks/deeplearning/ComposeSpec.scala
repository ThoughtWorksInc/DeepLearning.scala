package com.thoughtworks.deeplearning

import cats._
import com.thoughtworks.deeplearning.BpAny._
import com.thoughtworks.deeplearning.BpDouble._
import com.thoughtworks.deeplearning.Conversion._
import com.thoughtworks.deeplearning.Poly.MathOps
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import org.scalatest.{FreeSpec, Matchers}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class ComposeSpec extends FreeSpec with Matchers {
  "compose" in {
    def makeNetwork1(implicit x: shapeless.the.`Parameter[Double]`.Out) = {
      x + x
    }
    val network1 = makeNetwork1
    def makeNetwork2(implicit x: shapeless.the.`Parameter[Double]`.Out) = {
      log(x)
    }
    val network2 = makeNetwork2
    def makeNetwork3(implicit x: shapeless.the.`Parameter[Double]`.Out) = {
      network1.compose(network2)
    }
    val network3 = makeNetwork3
    network3.train(0.1)
  }
}
