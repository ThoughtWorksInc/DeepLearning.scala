package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.boolean._
import com.thoughtworks.deeplearning.array2D._
import com.thoughtworks.deeplearning.hlist._
import com.thoughtworks.deeplearning.any._
import com.thoughtworks.deeplearning.double._
import org.scalatest.{FreeSpec, Matchers}
import cats._

import scala.language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
class DifferentiableTypeSpec extends FreeSpec with Matchers {

  "Double :: HNil" in {
    "implicitly[(Double :: HNil) =:= Type[shapeless.::[Eval[scala.Double], shapeless.HNil], shapeless.:+:[Eval[scala.Double], shapeless.CNil]]]" should compile
    "implicitly[Array[Double :: HNil] =:= Array[Type[shapeless.::[Eval[scala.Double], shapeless.HNil], shapeless.:+:[Eval[scala.Double], shapeless.CNil]]]]" should compile
  }

  "x" in {
    val inputSymbol: Double :: HNil = implicitly

    implicitly[inputSymbol.Batch =:= (Double :: HNil)#Batch]
    implicitly[Layer.Aux[(Double :: HNil)#Batch, (Boolean :: HNil)#Batch] =:= inputSymbol.To[Boolean :: HNil]]

  }
}
