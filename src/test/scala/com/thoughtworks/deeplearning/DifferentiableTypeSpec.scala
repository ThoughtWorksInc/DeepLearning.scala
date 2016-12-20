package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.boolean._
import com.thoughtworks.deeplearning.array2D._
import com.thoughtworks.deeplearning.hlist._
import com.thoughtworks.deeplearning.dsl._
import com.thoughtworks.deeplearning.double._
import org.scalatest.{FreeSpec, Matchers}
import cats._

import scala.language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
class DifferentiableTypeSpec extends FreeSpec with Matchers {

  "BpDouble :**: BpHNil" in {
    "implicitly[(BpDouble :**: BpHNil) =:= BackPropagationType[shapeless.::[Eval[scala.Double], shapeless.HNil], shapeless.:+:[Eval[scala.Double], shapeless.CNil]]]" should compile
    "implicitly[Array[BpDouble :**: BpHNil] =:= Array[BackPropagationType[shapeless.::[Eval[scala.Double], shapeless.HNil], shapeless.:+:[Eval[scala.Double], shapeless.CNil]]]]" should compile
  }

  "x" in {
    val inputSymbol: BpDouble :**: BpHNil = implicitly

    implicitly[inputSymbol.Batch =:= (BpDouble :**: BpHNil)#Batch]
    implicitly[Layer.Aux[(BpDouble :**: BpHNil)#Batch, (BpBoolean :**: BpHNil)#Batch] =:= inputSymbol.To[BpBoolean :**: BpHNil]]

  }
}
