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
    "implicitly[DoubleBackProgationType.Batch <:< Batch.Aux[Double, Double]]" should compile
    "implicitly[DoubleBackProgationType.Batch =:= Batch.Aux[Double, Double]]" should compile
    "implicitly[DoubleBackProgationType.Batch <:< BackPropagationType[_, _]#Batch]" should compile
    "implicitly[DoubleBackProgationType.Batch =:= BackPropagationType[_, _]#Batch]" shouldNot compile
    "implicitly[BackPropagationType[_, _]#Batch =:= DoubleBackProgationType.Batch]" shouldNot compile
    "implicitly[(DoubleBackProgationType :**: BpHNil)#Batch <:< BpHList#Batch]" should compile
    "implicitly[(BpBoolean :**: DoubleBackProgationType :**: BpHNil)#Batch <:< BpHList#Batch]" should compile
    "implicitly[(DoubleBackProgationType :**: BpHNil)#Batch =:= Batch.Aux[Double :: shapeless.HNil, Double :+: CNil]]" should compile
    "implicitly[(DoubleBackProgationType :**: BpHNil)#Batch <:< BpHList#Batch]" should compile
    "implicitly[BpHList#Batch <:< (DoubleBackProgationType :**: BpHNil)#Batch]" shouldNot compile
    "implicitly[(BpAny :**: BpHNil)#Batch <:< BpHList#Batch]" should compile
    "implicitly[(BpAny :**: BpHList)#Batch <:< BpHList#Batch]" should compile
    "implicitly[(BpAny :**: BpHList)#Batch <:< (BpAny :**: BpHList)#Batch]" should compile
    "implicitly[(BpAny :**: BpHList)#Batch =:= (BpAny :**: BpHList)#Batch]" should compile
    "implicitly[(BpAny :**: BpHNil)#Batch =:= BpHList#Batch]" shouldNot compile
    "implicitly[(BpBoolean :**: DoubleBackProgationType :**: BpHNil) <:< BpHList]" should compile
    "implicitly[(BpBoolean :**: DoubleBackProgationType :**: BpHNil) <:< (BpBoolean :**: BpHList)]" shouldNot compile
  }

  "(BpAny :**: BpHList)#Batch" ignore {
    /*
      以下几个测试符合逻辑，但Scala编译器不认可
      没有很好的解决办法，只能尽量避免使用抽象类型吧
     */

    "implicitly[(DoubleBackProgationType :**: BpHNil)#Batch <:< (DoubleBackProgationType :**: BpHList)#Batch]" should compile
    "implicitly[(BpAny :**: BpHNil)#Data <:< BpHList#Data]" should compile
    "implicitly[(Double :: HNil) <:< BpHList#Data]" should compile
    "implicitly[(BpBoolean :**: DoubleBackProgationType :**: BpHNil)#Data <:< (BpBoolean :**: BpHList)#Data]" should compile
    "implicitly[(BpBoolean :**: DoubleBackProgationType :**: BpHNil)#Batch <:< (BpBoolean :**: BpHList)#Batch]" should compile
    "implicitly[(DoubleBackProgationType :**: BpHNil)#Batch <:< (BpAny :**: BpHNil)#Batch]" should compile
    "implicitly[(BpBoolean :**: DoubleBackProgationType :**: BpHNil)#Batch <:< (BpBoolean :**: BpAny :**: BpHNil)#Batch]" should compile
  }

}
