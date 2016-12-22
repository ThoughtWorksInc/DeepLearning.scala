package com.thoughtworks.deeplearning

import cats._
import com.thoughtworks.deeplearning.Conversion._
import com.thoughtworks.deeplearning.BpDouble._
import org.scalatest.{FreeSpec, Matchers}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class ToLayerSpec extends FreeSpec with Matchers {
  "ToLayer" in {
    """
    implicitly[
      ToLayer.OfType[Int, BpDouble#Batch, BpDouble] =:= ToLayer.Aux[Int, BpDouble#Batch, Double, Double]
    ]
    """ should compile

    """
    implicitly[
      ToLayer.Aux[Int, BpDouble#Batch, Double, Double] =:= ToLayer.OfType[Int, BpDouble#Batch, BpDouble]
    ]
    """ should compile
  }
}
