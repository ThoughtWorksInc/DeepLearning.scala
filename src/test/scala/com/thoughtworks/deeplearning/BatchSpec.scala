package com.thoughtworks.deeplearning

import cats.Eval
import org.scalatest.{FreeSpec, Matchers}
import com.thoughtworks.deeplearning.BpBoolean._
import com.thoughtworks.deeplearning.BpDouble._
import com.thoughtworks.deeplearning.BpAny._
import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.BpHList._
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
    "implicitly[(DoublePlaceholder :**: BpHNil)#Batch <:< BpHList#Batch]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: BpHNil)#Batch <:< BpHList#Batch]" should compile
    "implicitly[(DoublePlaceholder :**: BpHNil)#Batch =:= Batch.Aux[Double :: shapeless.HNil, Double :+: CNil]]" should compile
    "implicitly[(DoublePlaceholder :**: BpHNil)#Batch <:< BpHList#Batch]" should compile
    "implicitly[BpHList#Batch <:< (DoublePlaceholder :**: BpHNil)#Batch]" shouldNot compile
    "implicitly[(BpAny :**: BpHNil)#Batch <:< BpHList#Batch]" should compile
    "implicitly[(BpAny :**: BpHList)#Batch <:< BpHList#Batch]" should compile
    "implicitly[(BpAny :**: BpHList)#Batch <:< (BpAny :**: BpHList)#Batch]" should compile
    "implicitly[(BpAny :**: BpHList)#Batch =:= (BpAny :**: BpHList)#Batch]" should compile
    "implicitly[(BpAny :**: BpHNil)#Batch =:= BpHList#Batch]" shouldNot compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: BpHNil) <:< BpHList]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: BpHNil) <:< (BooleanPlaceholder :**: BpHList)]" shouldNot compile
  }

  "(BpAny :**: BpHList)#Batch" ignore {
    /*
      以下几个测试符合逻辑，但Scala编译器不认可
      没有很好的解决办法，只能尽量避免使用抽象类型吧
     */

    "implicitly[(DoublePlaceholder :**: BpHNil)#Batch <:< (DoublePlaceholder :**: BpHList)#Batch]" should compile
    "implicitly[(BpAny :**: BpHNil)#Data <:< BpHList#Data]" should compile
    "implicitly[(Double :: HNil) <:< BpHList#Data]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: BpHNil)#Data <:< (BooleanPlaceholder :**: BpHList)#Data]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: BpHNil)#Batch <:< (BooleanPlaceholder :**: BpHList)#Batch]" should compile
    "implicitly[(DoublePlaceholder :**: BpHNil)#Batch <:< (BpAny :**: BpHNil)#Batch]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: BpHNil)#Batch <:< (BooleanPlaceholder :**: BpAny :**: BpHNil)#Batch]" should compile
  }

}
