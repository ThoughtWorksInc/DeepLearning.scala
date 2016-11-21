package com.thoughtworks.deeplearning

import cats.Eval
import org.scalatest.{FreeSpec, Matchers}
import double._
import any._
import hlist._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class BatchSpec extends FreeSpec with Matchers {

  /*

  Batch有两种，一种是Invariant的，一种是covariant/contravariant的（即Widen版）。所有的Layer中应该使用后者

   */
  "Batch#Batch" in {
    "implicitly[Double#Batch <:< Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]" should compile
    "implicitly[Double#Batch =:= Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]" should compile
    "implicitly[Double#Batch <:< Any#Batch]" should compile
    "implicitly[Double#Batch =:= Any#Batch]" shouldNot compile
    "implicitly[Any#Batch =:= Double#Batch]" shouldNot compile
    "implicitly[(Double :: HNil)#Batch <:< HList#Batch]" should compile
    "implicitly[(Boolean :: Double :: HNil)#Batch <:< HList#Batch]" should compile
    "implicitly[(Double :: HNil)#Batch =:= Batch.Aux[shapeless.::[Eval[scala.Double], shapeless.HNil], shapeless.:+:[Eval[scala.Double], shapeless.CNil]]]" should compile
    "implicitly[(Double :: HNil)#Batch <:< HList#Batch]" should compile
    "implicitly[HList#Batch <:< (Double :: HNil)#Batch]" shouldNot compile
    "implicitly[(Any :: HNil)#Batch <:< HList#Batch]" should compile
    "implicitly[(Any :: HList)#Batch <:< HList#Batch]" should compile
    "implicitly[(Any :: HList)#Batch <:< (Any :: HList)#Batch]" should compile
    "implicitly[(Any :: HList)#Batch =:= (Any :: HList)#Batch]" should compile
    "implicitly[(Any :: HNil)#Batch =:= HList#Batch]" shouldNot compile
    "implicitly[(Boolean :: Double :: HNil) <:< HList]" should compile
    "implicitly[(Boolean :: Double :: HNil) <:< (Boolean :: HList)]" shouldNot compile
  }

  "(Any :: HList)#Batch" ignore {
    /*
      以下几个测试符合逻辑，但Scala编译器不认可
      没有很好的解决办法，只能尽量避免使用抽象类型吧
     */

    "implicitly[(Double :: HNil)#Batch <:< (Double :: HList)#Batch]" should compile
    "implicitly[(Any :: HNil)#Data <:< HList#Data]" should compile
    "implicitly[shapeless.::[cats.Eval[Double],shapeless.HNil] <:< HList#Data]" should compile
    "implicitly[(Boolean :: Double :: HNil)#Data <:< (Boolean :: HList)#Data]" should compile
    "implicitly[(Boolean :: Double :: HNil)#Batch <:< (Boolean :: HList)#Batch]" should compile
    "implicitly[(Double :: HNil)#Batch <:< (Any :: HNil)#Batch]" should compile
    "implicitly[(Boolean :: Double :: HNil)#Batch <:< (Boolean :: Any :: HNil)#Batch]" should compile
  }

}
