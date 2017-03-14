package com.thoughtworks.deeplearning

import org.scalatest.{FreeSpec, Matchers}
import com.thoughtworks.enableMembersIf

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
@enableMembersIf(!scala.util.Properties.versionNumberString.startsWith("2.12."))
final class Issue8 extends FreeSpec with Matchers {

  import com.thoughtworks.deeplearning.DifferentiableINDArray._
  import com.thoughtworks.deeplearning.DifferentiableAny._
  import com.thoughtworks.deeplearning.Layer.Tape
  import com.thoughtworks.deeplearning.Symbolic._
  import com.thoughtworks.deeplearning.Poly.MathFunctions._
  import com.thoughtworks.deeplearning.Poly.MathOps
  import org.nd4j.linalg.api.ndarray.INDArray
  import org.nd4s.Implicits._
  import shapeless._

  "issue8" in {
    def layer(implicit x: From[INDArray]##`@`) = {
      val x1 = (-x).withOutputDataHook { x: INDArray =>
        println(x)
      }
      x1.sum(1) / x
    }

    layer.train(Array(Array(1, 2, 3, 4), Array(1, 2, 3, 4), Array(1, 2, 3, 4), Array(1, 2, 3, 4)).toNDArray)
  }

}
