package com.thoughtworks.deeplearning

import shapeless._
import com.thoughtworks.deeplearning.BpBoolean._
import com.thoughtworks.deeplearning.Bp2DArray._
import com.thoughtworks.deeplearning.BpHList._
import com.thoughtworks.deeplearning.Conversion._
import com.thoughtworks.deeplearning.BpDouble._
import org.scalatest.{FreeSpec, Matchers}
import cats._

import language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
class DifferentiableTypeSpec extends FreeSpec with Matchers {

  "DoubleBackProgationType :**: BpHNil" in {
    "implicitly[(DoubleBackProgationType :**: BpHNil) =:= BackPropagationType[::[Double, shapeless.HNil], shapeless.:+:[Double, shapeless.CNil]]]" should compile
    "implicitly[Array[DoubleBackProgationType :**: BpHNil] =:= Array[BackPropagationType[::[Double, shapeless.HNil], shapeless.:+:[Double, shapeless.CNil]]]]" should compile
  }

  "x" in {
    val inputSymbol: DoubleBackProgationType :**: BpHNil = implicitly

    implicitly[inputSymbol.Batch =:= (DoubleBackProgationType :**: BpHNil)#Batch]
    implicitly[Layer.Aux[(DoubleBackProgationType :**: BpHNil)#Batch, (BpBoolean :**: BpHNil)#Batch] =:= inputSymbol.To[BpBoolean :**: BpHNil]]

  }
}
