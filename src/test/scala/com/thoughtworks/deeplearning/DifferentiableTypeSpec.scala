package com.thoughtworks.deeplearning

import shapeless._
import com.thoughtworks.deeplearning.DifferentiableBoolean._
import com.thoughtworks.deeplearning.DifferentiableHList._
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import org.scalatest.{FreeSpec, Matchers}
import cats._

import language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
class DifferentiableTypeSpec extends FreeSpec with Matchers {

  "DoublePlaceholder :**: HNilPlaceholder" in {
    "implicitly[(DoublePlaceholder :**: HNilPlaceholder) =:= Placeholder[::[Double, shapeless.HNil], shapeless.:+:[Double, shapeless.CNil]]]" should compile
    "implicitly[Array[DoublePlaceholder :**: HNilPlaceholder] =:= Array[Placeholder[::[Double, shapeless.HNil], shapeless.:+:[Double, shapeless.CNil]]]]" should compile
  }

  "x" in {
    val inputSymbol: DoublePlaceholder :**: HNilPlaceholder = implicitly

    implicitly[inputSymbol.Tape =:= (DoublePlaceholder :**: HNilPlaceholder)#Tape]
    implicitly[Layer.Aux[
      (DoublePlaceholder :**: HNilPlaceholder)#Tape,
      (BooleanPlaceholder :**: HNilPlaceholder)#Tape] =:= inputSymbol.To[BooleanPlaceholder :**: HNilPlaceholder]]

  }
}
