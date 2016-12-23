package com.thoughtworks.deeplearning

import cats.Eval
import org.scalatest.{FreeSpec, Matchers}
import com.thoughtworks.deeplearning.DifferentiableBoolean._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.DifferentiableHList._
import language.existentials
import shapeless._
import com.thoughtworks.deeplearning.Layer.Batch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class BatchSpec extends FreeSpec with Matchers {

  /*

  Batch有两种，一种是Invariant的，一种是covariant/contravariant的（即Widen版）。所有的Layer中应该使用后者

   */
  "Batch#Batch" in {
    "implicitly[DoublePlaceholder.Batch <:< Batch.Aux[Double, Double]]" should compile
    "implicitly[DoublePlaceholder.Batch =:= Batch.Aux[Double, Double]]" should compile
    "implicitly[DoublePlaceholder.Batch <:< Placeholder[_, _]#Batch]" should compile
    "implicitly[DoublePlaceholder.Batch =:= Placeholder[_, _]#Batch]" shouldNot compile
    "implicitly[Placeholder[_, _]#Batch =:= DoublePlaceholder.Batch]" shouldNot compile
    "implicitly[(DoublePlaceholder :**: DifferentiableHNil)#Batch <:< DifferentiableHList#Batch]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: DifferentiableHNil)#Batch <:< DifferentiableHList#Batch]" should compile
    "implicitly[(DoublePlaceholder :**: DifferentiableHNil)#Batch =:= Batch.Aux[Double :: shapeless.HNil, Double :+: CNil]]" should compile
    "implicitly[(DoublePlaceholder :**: DifferentiableHNil)#Batch <:< DifferentiableHList#Batch]" should compile
    "implicitly[DifferentiableHList#Batch <:< (DoublePlaceholder :**: DifferentiableHNil)#Batch]" shouldNot compile
    "implicitly[(AnyPlaceholder :**: DifferentiableHNil)#Batch <:< DifferentiableHList#Batch]" should compile
    "implicitly[(AnyPlaceholder :**: DifferentiableHList)#Batch <:< DifferentiableHList#Batch]" should compile
    "implicitly[(AnyPlaceholder :**: DifferentiableHList)#Batch <:< (AnyPlaceholder :**: DifferentiableHList)#Batch]" should compile
    "implicitly[(AnyPlaceholder :**: DifferentiableHList)#Batch =:= (AnyPlaceholder :**: DifferentiableHList)#Batch]" should compile
    "implicitly[(AnyPlaceholder :**: DifferentiableHNil)#Batch =:= DifferentiableHList#Batch]" shouldNot compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: DifferentiableHNil) <:< DifferentiableHList]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: DifferentiableHNil) <:< (BooleanPlaceholder :**: DifferentiableHList)]" shouldNot compile
  }

  "(AnyPlaceholder :**: DifferentiableHList)#Batch" ignore {
    /*
      以下几个测试符合逻辑，但Scala编译器不认可
      没有很好的解决办法，只能尽量避免使用抽象类型吧
     */

    "implicitly[(DoublePlaceholder :**: DifferentiableHNil)#Batch <:< (DoublePlaceholder :**: DifferentiableHList)#Batch]" should compile
    "implicitly[(AnyPlaceholder :**: DifferentiableHNil)#Data <:< DifferentiableHList#Data]" should compile
    "implicitly[(Double :: HNil) <:< DifferentiableHList#Data]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: DifferentiableHNil)#Data <:< (BooleanPlaceholder :**: DifferentiableHList)#Data]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: DifferentiableHNil)#Batch <:< (BooleanPlaceholder :**: DifferentiableHList)#Batch]" should compile
    "implicitly[(DoublePlaceholder :**: DifferentiableHNil)#Batch <:< (AnyPlaceholder :**: DifferentiableHNil)#Batch]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: DifferentiableHNil)#Batch <:< (BooleanPlaceholder :**: AnyPlaceholder :**: DifferentiableHNil)#Batch]" should compile
  }

}
