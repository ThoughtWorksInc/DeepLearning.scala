//package com.thoughtworks.deepLearning
//
//import cats.Eval
//import com.thoughtworks.deepLearning.any.InputAst
//import com.thoughtworks.deepLearning.double.maxDoubleDouble
//import com.thoughtworks.deepLearning.double._
//import com.thoughtworks.deepLearning.double.Double
//
//object DoubleSpec {
////
////  object Differentiable {
////    type Batch[+Data0, -Delta0] = Differentiable {
////      type Data <: Data0
////      type Delta >: Delta0
////    }
////  }
////
////  trait Differentiable extends AutoCloseable { outer =>
////    type Data
////    type Delta
////
////    type Covariant >: Differentiable.Covariant[Data, Delta] <: Differentiable.Covariant[Data, Delta]
////
////  }
////  type Double = Differentiable {
////    type Delta = Eval[scala.Double]
////    type Data = Eval[scala.Double]
////  }
////
////  final case class Identity[Input0 <: Differentiable]()
////
////  type InputAst[InputTypePair <: Differentiable] = Identity[InputTypePair#Batch]
////
////  def maxDoubleDouble[Input <: Differentiable](id: Identity[Input]) = ???
//
//  def crash(implicit x: InputAst[Double]) = {
//    max(1.0 - 0.0.toLiteral, 0.0)
//  }
//}
