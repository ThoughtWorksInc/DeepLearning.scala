package com.thoughtworks.deeplearning

import cats._
import com.thoughtworks.deeplearning.double._
import com.thoughtworks.deeplearning.dsl._
import org.scalatest.{FreeSpec, Matchers}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class ComposeSpec extends FreeSpec with Matchers {
  "compose" in {
    def makeNetwork1(implicit x: Double) = {
      x + x
    }
    val network1 = makeNetwork1
    def makeNetwork2(implicit x: Double) = {
      log(x)
    }
    val network2 = makeNetwork2
    def makeNetwork3(implicit x: Double) = {
      network1.compose(network2)
    }
    val network3 = makeNetwork3
    network3.train(Eval.now(0.1))
  }
}
