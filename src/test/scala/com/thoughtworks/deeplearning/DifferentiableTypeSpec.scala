package com.thoughtworks.deeplearning

import shapeless._
import com.thoughtworks.deeplearning.DifferentiableBoolean._
import com.thoughtworks.deeplearning.DifferentiableINDArray._
import com.thoughtworks.deeplearning.DifferentiableHList._
import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import org.scalatest.{FreeSpec, Matchers}
import cats._

import language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
class DifferentiableTypeSpec extends FreeSpec with Matchers {

  "DoublePlaceholder :**: DifferentiableHNil" in {
    "implicitly[(DoublePlaceholder :**: DifferentiableHNil) =:= Placeholder[::[Double, shapeless.HNil], shapeless.:+:[Double, shapeless.CNil]]]" should compile
    "implicitly[Array[DoublePlaceholder :**: DifferentiableHNil] =:= Array[Placeholder[::[Double, shapeless.HNil], shapeless.:+:[Double, shapeless.CNil]]]]" should compile
  }

  "x" in {
    val inputSymbol: DoublePlaceholder :**: DifferentiableHNil = implicitly

    implicitly[inputSymbol.Batch =:= (DoublePlaceholder :**: DifferentiableHNil)#Batch]
    implicitly[Layer.Aux[(DoublePlaceholder :**: DifferentiableHNil)#Batch, (BooleanPlaceholder :**: DifferentiableHNil)#Batch] =:= inputSymbol.To[BooleanPlaceholder :**: DifferentiableHNil]]

  }
}
