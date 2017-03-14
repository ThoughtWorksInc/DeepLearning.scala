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
import com.thoughtworks.deeplearning.Layer.Tape

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class TapeSpec extends FreeSpec with Matchers {

  /*

  Tape有两种，一种是Invariant的，一种是covariant/contravariant的（即Widen版）。所有的Layer中应该使用后者

   */
  "Tape#Tape" in {
    "implicitly[DoublePlaceholder.Tape <:< Tape.Aux[Double, Double]]" should compile
    "implicitly[DoublePlaceholder.Tape =:= Tape.Aux[Double, Double]]" should compile
    "implicitly[DoublePlaceholder.Tape <:< Placeholder[_, _]#Tape]" should compile
    "implicitly[DoublePlaceholder.Tape =:= Placeholder[_, _]#Tape]" shouldNot compile
    "implicitly[Placeholder[_, _]#Tape =:= DoublePlaceholder.Tape]" shouldNot compile
    "implicitly[(DoublePlaceholder :**: HNilPlaceholder)#Tape <:< HListPlaceholder#Tape]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: HNilPlaceholder)#Tape <:< HListPlaceholder#Tape]" should compile
    "implicitly[(DoublePlaceholder :**: HNilPlaceholder)#Tape =:= Tape.Aux[Double :: shapeless.HNil, Double :+: CNil]]" should compile
    "implicitly[(DoublePlaceholder :**: HNilPlaceholder)#Tape <:< HListPlaceholder#Tape]" should compile
    "implicitly[HListPlaceholder#Tape <:< (DoublePlaceholder :**: HNilPlaceholder)#Tape]" shouldNot compile
    "implicitly[(AnyPlaceholder :**: HNilPlaceholder)#Tape <:< HListPlaceholder#Tape]" should compile
    "implicitly[(AnyPlaceholder :**: HListPlaceholder)#Tape <:< HListPlaceholder#Tape]" should compile
    "implicitly[(AnyPlaceholder :**: HListPlaceholder)#Tape <:< (AnyPlaceholder :**: HListPlaceholder)#Tape]" should compile
    "implicitly[(AnyPlaceholder :**: HListPlaceholder)#Tape =:= (AnyPlaceholder :**: HListPlaceholder)#Tape]" should compile
    "implicitly[(AnyPlaceholder :**: HNilPlaceholder)#Tape =:= HListPlaceholder#Tape]" shouldNot compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: HNilPlaceholder) <:< HListPlaceholder]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: HNilPlaceholder) <:< (BooleanPlaceholder :**: HListPlaceholder)]" shouldNot compile
  }

  "(AnyPlaceholder :**: HListPlaceholder)#Tape" ignore {
    /*
      以下几个测试符合逻辑，但Scala编译器不认可
      没有很好的解决办法，只能尽量避免使用抽象类型吧
     */

    "implicitly[(DoublePlaceholder :**: HNilPlaceholder)#Tape <:< (DoublePlaceholder :**: HListPlaceholder)#Tape]" should compile
    "implicitly[(AnyPlaceholder :**: HNilPlaceholder)#Data <:< HListPlaceholder#Data]" should compile
    "implicitly[(Double :: HNil) <:< HListPlaceholder#Data]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: HNilPlaceholder)#Data <:< (BooleanPlaceholder :**: HListPlaceholder)#Data]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: HNilPlaceholder)#Tape <:< (BooleanPlaceholder :**: HListPlaceholder)#Tape]" should compile
    "implicitly[(DoublePlaceholder :**: HNilPlaceholder)#Tape <:< (AnyPlaceholder :**: HNilPlaceholder)#Tape]" should compile
    "implicitly[(BooleanPlaceholder :**: DoublePlaceholder :**: HNilPlaceholder)#Tape <:< (BooleanPlaceholder :**: AnyPlaceholder :**: HNilPlaceholder)#Tape]" should compile
  }

}
