package com.thoughtworks.deepLearning

import cats._
import com.thoughtworks.deepLearning.any.ToNeuralNetwork
import com.thoughtworks.deepLearning.double._
import org.scalatest.{FreeSpec, Matchers}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class ToAstSpec extends FreeSpec with Matchers {
  "ToNeuralNetwork" in {
    implicitly[
      ToNeuralNetwork.OfType[Int, Double#Batch, Double] =:= ToNeuralNetwork.Aux[Int, Double#Batch, Eval[scala.Double], Eval[scala.Double]]]

    implicitly[
      ToNeuralNetwork.Aux[Int, Double#Batch, Eval[scala.Double], Eval[scala.Double]] =:= ToNeuralNetwork.OfType[Int, Double#Batch, Double]]
  }
}
