package com.thoughtworks.deeplearning

import cats._
import cats.implicits._
import com.thoughtworks.deeplearning.BpDouble._
import com.thoughtworks.deeplearning.Conversion._
import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning._
import org.scalatest._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DoubleToLiteralSpec extends FreeSpec with Matchers {
  "the ToLiteral should be a double batch" in {
    """implicitly[shapeless.the.`ToLiteral[Double]`.Out <:< Batch.Aux[Eval[Double], Eval[Double]]]""" should compile
  }
}
