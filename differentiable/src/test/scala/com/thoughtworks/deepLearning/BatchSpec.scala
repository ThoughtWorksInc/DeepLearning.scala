package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.Ast._
import com.thoughtworks.deepLearning.Batch._
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

  Batch有两种，一种是Invariant的，一种是covariant/contravariant的（即Widen版）。所有的Ast中应该使用后者

   */
  "Batch#Widen" in {
    "implicitly[Double#Widen <:< WidenBatch[Eval[scala.Double], Eval[scala.Double]]]" should compile
    "implicitly[Double#Widen =:= WidenBatch[Eval[scala.Double], Eval[scala.Double]]]" should compile
    "implicitly[Double#Widen <:< Any#Widen]" should compile
    "implicitly[Double#Widen =:= Any#Widen]" shouldNot compile
    "implicitly[Any#Widen =:= Double#Widen]" shouldNot compile
    "implicitly[(Double :: HNil)#Widen <:< HList#Widen]" should compile
    "implicitly[(Boolean :: Double :: HNil)#Widen <:< HList#Widen]" should compile
    "implicitly[(Double :: HNil)#Widen =:= WidenBatch[shapeless.::[Eval[scala.Double], shapeless.HNil], shapeless.:+:[Eval[scala.Double], shapeless.CNil]]]" should compile
    "implicitly[(Any :: HNil)#Widen <:< HList#Widen]" should compile
    "implicitly[(Any :: HList)#Widen <:< HList#Widen]" should compile
    "implicitly[(Any :: HList)#Widen <:< (Any :: HList)#Widen]" should compile
    "implicitly[(Any :: HList)#Widen =:= (Any :: HList)#Widen]" should compile
    "implicitly[(Any :: HNil)#Widen =:= HList#Widen]" shouldNot compile
    "implicitly[(Boolean :: Double :: HNil) <:< HList]" should compile
    "implicitly[(Boolean :: Double :: HNil) <:< (Boolean :: HList)]" shouldNot compile
  }

  "(Any :: HList)#Widen" ignore {
    /*
      以下几个测试符合逻辑，但Scala编译器不认可
      */


    "implicitly[(Any :: HNil)#Data <:< HList#Data]" should compile
    "implicitly[shapeless.::[cats.Eval[Double],shapeless.HNil] <:< HList#Data]" should compile
    "implicitly[(Boolean :: Double :: HNil)#Data <:< (Boolean :: HList)#Data]" should compile
    "implicitly[(Boolean :: Double :: HNil)#Widen <:< (Boolean :: HList)#Widen]" should compile
    "implicitly[(Double :: HNil)#Widen <:< (Any :: HNil)#Widen]" should compile
    "implicitly[(Boolean :: Double :: HNil)#Widen <:< (Boolean :: Any :: HNil)#Widen]" should compile
  }

}
