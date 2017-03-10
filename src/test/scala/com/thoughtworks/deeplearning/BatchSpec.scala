package com.thoughtworks.deeplearning

import cats.Eval
import org.scalatest.{FreeSpec, Matchers}
import com.thoughtworks.deeplearning.DifferentiableBoolean._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.Symbolic._
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
    "implicitly[(DoublePlaceholder :**: HNilPlaceholder)#Batch <:< HListPlaceholder#Batch]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: HNilPlaceholder)#Batch <:< HListPlaceholder#Batch]" should compile
    "implicitly[(DoublePlaceholder :**: HNilPlaceholder)#Batch =:= Batch.Aux[Double :: shapeless.HNil, Double :+: CNil]]" should compile
    "implicitly[(DoublePlaceholder :**: HNilPlaceholder)#Batch <:< HListPlaceholder#Batch]" should compile
    "implicitly[HListPlaceholder#Batch <:< (DoublePlaceholder :**: HNilPlaceholder)#Batch]" shouldNot compile
    "implicitly[(AnyPlaceholder :**: HNilPlaceholder)#Batch <:< HListPlaceholder#Batch]" should compile
    "implicitly[(AnyPlaceholder :**: HListPlaceholder)#Batch <:< HListPlaceholder#Batch]" should compile
    "implicitly[(AnyPlaceholder :**: HListPlaceholder)#Batch <:< (AnyPlaceholder :**: HListPlaceholder)#Batch]" should compile
    "implicitly[(AnyPlaceholder :**: HListPlaceholder)#Batch =:= (AnyPlaceholder :**: HListPlaceholder)#Batch]" should compile
    "implicitly[(AnyPlaceholder :**: HNilPlaceholder)#Batch =:= HListPlaceholder#Batch]" shouldNot compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: HNilPlaceholder) <:< HListPlaceholder]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: HNilPlaceholder) <:< (BooleanPlaceholder :**: HListPlaceholder)]" shouldNot compile
  }

  "(AnyPlaceholder :**: HListPlaceholder)#Batch" ignore {
    /*
      以下几个测试符合逻辑，但Scala编译器不认可
      没有很好的解决办法，只能尽量避免使用抽象类型吧
     */

    "implicitly[(DoublePlaceholder :**: HNilPlaceholder)#Batch <:< (DoublePlaceholder :**: HListPlaceholder)#Batch]" should compile
    "implicitly[(AnyPlaceholder :**: HNilPlaceholder)#Data <:< HListPlaceholder#Data]" should compile
    "implicitly[(Double :: HNil) <:< HListPlaceholder#Data]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: HNilPlaceholder)#Data <:< (BooleanPlaceholder :**: HListPlaceholder)#Data]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: HNilPlaceholder)#Batch <:< (BooleanPlaceholder :**: HListPlaceholder)#Batch]" should compile
    "implicitly[(DoublePlaceholder :**: HNilPlaceholder)#Batch <:< (AnyPlaceholder :**: HNilPlaceholder)#Batch]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: HNilPlaceholder)#Batch <:< (BooleanPlaceholder :**: AnyPlaceholder :**: HNilPlaceholder)#Batch]" should compile
  }

}
