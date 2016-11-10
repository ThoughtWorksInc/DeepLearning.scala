package com.thoughtworks.deepLearning

import cats._
import com.thoughtworks.deepLearning.boolean._
import com.thoughtworks.deepLearning.array2D._
import com.thoughtworks.deepLearning.hlist._
import com.thoughtworks.deepLearning.double._
import org.scalatest.{FreeSpec, Matchers}
import Differentiable.Batch
import com.thoughtworks.deepLearning.DifferentiableFunction.Ast

import scala.language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
class DifferentiableTypeSpec extends FreeSpec with Matchers {

  "Double :: HNil" in {
    "implicitly[(Double :: HNil) =:= DifferentiableType.ConcreteType[shapeless.::[Eval[scala.Double], shapeless.HNil], shapeless.:+:[Eval[scala.Double], shapeless.CNil]]]" should compile
    "implicitly[Array[Double :: HNil] =:= Array[DifferentiableType.ConcreteType[shapeless.::[Eval[scala.Double], shapeless.HNil], shapeless.:+:[Eval[scala.Double], shapeless.CNil]]]]" should compile
  }

  "x" in {
    val inputSymbol: Double :: HNil = implicitly

    implicitly[inputSymbol.Batch =:= (Double :: HNil)#Batch]
    implicitly[Ast[(Double :: HNil)#Batch, (Boolean :: HNil)#Batch] =:= inputSymbol.Ast[Boolean :: HNil]]

  }
}
