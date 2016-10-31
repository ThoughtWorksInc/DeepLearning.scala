//package com.thoughtworks.deepLearning
//
//import cats.Eval
//import com.thoughtworks.deepLearning.dsl._
//
//
///**
//  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
//  */
//final class AnySpec {
//
//  val a: List[Any#Data] = ???
//
//  type BatchOf[A <: Any] = Differentiable.Batch.Aux[A#Data, A#Delta]
//
//  def cast(x: Double#Batch): Any#Batch = x
//  def cast2(x: HNil#Batch): HList#Batch = x
//  def cast3(x: (Any :: Double :: HNil)#Batch): HList#Batch = x
//  def cast4(x: (HList :: HNil)#Batch)
//    : Batch.Aux[shapeless.::[shapeless.HList, shapeless.HNil], shapeless.:+:[scala.Nothing, shapeless.CNil]] = x
//  def cast5(x: (Double :: HNil)#Batch)
//    : Batch.Aux[shapeless.::[Eval[scala.Double], shapeless.HNil], shapeless.:+:[Eval[scala.Double], shapeless.CNil]] =
//    x
//  def cast6(
//      x: Batch.Aux[shapeless.::[Eval[scala.Double], shapeless.HNil],
//                   shapeless.:+:[Eval[scala.Double], shapeless.CNil]]): (Double :: HNil)#Batch = x
//  def cast7(
//      x: Batch.Aux[shapeless.::[Eval[scala.Double], shapeless.HNil],
//                   shapeless.:+:[Eval[scala.Double], shapeless.CNil]]): BatchOf[Double :: HNil] = x
//  def cast8(x: BatchOf[Double :: HNil])
//    : Batch.Aux[shapeless.::[Eval[scala.Double], shapeless.HNil], shapeless.:+:[Eval[scala.Double], shapeless.CNil]] =
//    x
//  //  def cast3(x: HList#AbstractBatch): HList#Batch = x
////  def cast4(x: HList#Batch): HList#AbstractBatch = x
//
////  val x: Batch.Aux[Double#Data, Double#Delta] <:< Batch.Aux[Any#Data, Any#Delta] = implicitly
//
//}
