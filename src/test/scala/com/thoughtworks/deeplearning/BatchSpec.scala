package com.thoughtworks.deeplearning

import cats.Eval
import org.scalatest.{FreeSpec, Matchers}
import BpBoolean._
import double._
import com.thoughtworks.deeplearning.BpAny._
import com.thoughtworks.deeplearning.Conversion._
import hlist._
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
    "implicitly[BpDouble#Batch <:< Batch.Aux[Eval[Double], Eval[Double]]]" should compile
    "implicitly[BpDouble#Batch =:= Batch.Aux[Eval[Double], Eval[Double]]]" should compile
    "implicitly[BpDouble#Batch <:< BackPropagationType[_, _]#Batch]" should compile
    "implicitly[BpDouble#Batch =:= BackPropagationType[_, _]#Batch]" shouldNot compile
    "implicitly[BackPropagationType[_, _]#Batch =:= BpDouble#Batch]" shouldNot compile
    "implicitly[(BpDouble :**: BpHNil)#Batch <:< BpHList#Batch]" should compile
    "implicitly[(BpBoolean :**: BpDouble :**: BpHNil)#Batch <:< BpHList#Batch]" should compile
    "implicitly[(BpDouble :**: BpHNil)#Batch =:= Batch.Aux[Eval[Double] :: shapeless.HNil, Eval[Double] :+: CNil]]" should compile
    "implicitly[(BpDouble :**: BpHNil)#Batch <:< BpHList#Batch]" should compile
    "implicitly[BpHList#Batch <:< (BpDouble :**: BpHNil)#Batch]" shouldNot compile
    "implicitly[(BpAny :**: BpHNil)#Batch <:< BpHList#Batch]" should compile
    "implicitly[(BpAny :**: BpHList)#Batch <:< BpHList#Batch]" should compile
    "implicitly[(BpAny :**: BpHList)#Batch <:< (BpAny :**: BpHList)#Batch]" should compile
    "implicitly[(BpAny :**: BpHList)#Batch =:= (BpAny :**: BpHList)#Batch]" should compile
    "implicitly[(BpAny :**: BpHNil)#Batch =:= BpHList#Batch]" shouldNot compile
    "implicitly[(BpBoolean :**: BpDouble :**: BpHNil) <:< BpHList]" should compile
    "implicitly[(BpBoolean :**: BpDouble :**: BpHNil) <:< (BpBoolean :**: BpHList)]" shouldNot compile
  }

  "(BpAny :**: BpHList)#Batch" ignore {
    /*
      以下几个测试符合逻辑，但Scala编译器不认可
      没有很好的解决办法，只能尽量避免使用抽象类型吧
     */

    "implicitly[(BpDouble :**: BpHNil)#Batch <:< (BpDouble :**: BpHList)#Batch]" should compile
    "implicitly[(BpAny :**: BpHNil)#Data <:< BpHList#Data]" should compile
    "implicitly[(cats.Eval[BpDouble] :: HNil) <:< BpHList#Data]" should compile
    "implicitly[(BpBoolean :**: BpDouble :**: BpHNil)#Data <:< (BpBoolean :**: BpHList)#Data]" should compile
    "implicitly[(BpBoolean :**: BpDouble :**: BpHNil)#Batch <:< (BpBoolean :**: BpHList)#Batch]" should compile
    "implicitly[(BpDouble :**: BpHNil)#Batch <:< (BpAny :**: BpHNil)#Batch]" should compile
    "implicitly[(BpBoolean :**: BpDouble :**: BpHNil)#Batch <:< (BpBoolean :**: BpAny :**: BpHNil)#Batch]" should compile
  }

}
