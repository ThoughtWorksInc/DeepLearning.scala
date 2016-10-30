package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.dsl._
import com.thoughtworks.deepLearning.NeuralNetwork.Batch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class AnySpec {

  val a: List[Any#Data] = ???
//
  def cast(x: Double#Batch): Any#Batch = x
  def cast2(x: HNil#Batch): HList#Batch = x
  def cast3(x: (Any :: Double :: HNil)#Batch): HList#Batch = x
  def cast4(x: (HList :: HNil)#Batch)
    : Batch.Aux[shapeless.::[shapeless.HList, shapeless.HNil], shapeless.:+:[scala.Nothing, shapeless.CNil]] = x
  def cast5(x: (Double :: HNil)#Batch)
    : Batch.Aux[shapeless.::[Eval[scala.Double], shapeless.HNil], shapeless.:+:[Eval[scala.Double], shapeless.CNil]] =
    x
  def cast6(
      x: Batch.Aux[shapeless.::[Eval[scala.Double], shapeless.HNil],
                   shapeless.:+:[Eval[scala.Double], shapeless.CNil]]): (Double :: HNil)#Batch = x
//  def cast3(x: HList#AbstractBatch): HList#Batch = x
//  def cast4(x: HList#Batch): HList#AbstractBatch = x

//  val x: Batch.Aux[Double#Data, Double#Delta] <:< Batch.Aux[Any#Data, Any#Delta] = implicitly

}
