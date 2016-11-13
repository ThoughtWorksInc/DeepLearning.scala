package com.thoughtworks.deepLearning

import cats._
import com.thoughtworks.deepLearning.boolean._
import com.thoughtworks.deepLearning.array2D._
import com.thoughtworks.deepLearning.hlist._
import com.thoughtworks.deepLearning.double._
import org.scalatest.{FreeSpec, Matchers}
import Batch.Aux
import com.thoughtworks.deepLearning.NeuralNetwork.Aux

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
    implicitly[NeuralNetwork.Aux[(Double :: HNil)#Batch, (Boolean :: HNil)#Batch] =:= inputSymbol.To[Boolean :: HNil]]

  }
}
