package com.thoughtworks.deeplearning

import shapeless._
import com.thoughtworks.deeplearning.BpBoolean._
import com.thoughtworks.deeplearning.Bp2DArray._
import com.thoughtworks.deeplearning.BpHList._
import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.BpDouble._
import org.scalatest.{FreeSpec, Matchers}
import cats._

import language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
class DifferentiableTypeSpec extends FreeSpec with Matchers {

  "DoublePlaceholder :**: BpHNil" in {
    "implicitly[(DoublePlaceholder :**: BpHNil) =:= Placeholder[::[Double, shapeless.HNil], shapeless.:+:[Double, shapeless.CNil]]]" should compile
    "implicitly[Array[DoublePlaceholder :**: BpHNil] =:= Array[Placeholder[::[Double, shapeless.HNil], shapeless.:+:[Double, shapeless.CNil]]]]" should compile
  }

  "x" in {
    val inputSymbol: DoublePlaceholder :**: BpHNil = implicitly

    implicitly[inputSymbol.Batch =:= (DoublePlaceholder :**: BpHNil)#Batch]
    implicitly[Layer.Aux[(DoublePlaceholder :**: BpHNil)#Batch, (BooleanPlaceholder :**: BpHNil)#Batch] =:= inputSymbol.To[BooleanPlaceholder :**: BpHNil]]

  }
}
