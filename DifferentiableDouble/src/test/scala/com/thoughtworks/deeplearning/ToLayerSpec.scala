package com.thoughtworks.deeplearning

import cats._
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import org.scalatest.{FreeSpec, Matchers}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class ToLayerSpec extends FreeSpec with Matchers {
  "ToLayer" in {
    """
    implicitly[
      ToLayer.OfPlaceholder[Int, DoublePlaceholder.Tape, DoublePlaceholder] =:= ToLayer.Aux[Int, DoublePlaceholder.Tape, Double, Double]
    ]
    """ should compile

    """
    implicitly[
      ToLayer.Aux[Int, DoublePlaceholder.Tape, Double, Double] =:= ToLayer.OfPlaceholder[Int, DoublePlaceholder.Tape, DoublePlaceholder]
    ]
    """ should compile
  }
}
