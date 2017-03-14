package com.thoughtworks.deeplearning

import cats._
import cats.implicits._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.Layer.Tape
import com.thoughtworks.deeplearning._
import org.scalatest._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class LiftDoubleSpec extends FreeSpec with Matchers {
  "ToLiteral[Double] should be a double tape" in {
    """implicitly[shapeless.the.`ToLiteral[Double]`.`@` <:< Tape.Aux[Double, Double]]""" should compile
  }

  "<=> should create Layers" in {
    """implicitly[shapeless.the.`Double <=> Double`.`@` =:= Layer.Aux[Tape.Aux[Double, Double], Tape.Aux[Double, Double]]]""" should compile
  }
}
