package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.Poly._
import com.thoughtworks.deeplearning.BpDouble._
import org.scalatest.{FreeSpec, Matchers}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class LiftSpec extends FreeSpec with Matchers {

  def plus(implicit x: shapeless.the.`Parameter[Double]`.Out): shapeless.the.`LayerOf[Double, Double]`.Out = {
    x + 1.0
  }

}

