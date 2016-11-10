package com.thoughtworks.deepLearning

import cats._
import cats.implicits._
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.scalatest.{FreeSpec, Matchers}
import double._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class ToAstSpec extends FreeSpec with Matchers {
  "ToAst" in {
    implicitly[
      ToAst.OfType[Int, Double#Batch, Double] =:= ToAst.Aux[Int, Double#Batch, Eval[scala.Double], Eval[scala.Double]]]

    implicitly[
      ToAst.Aux[Int, Double#Batch, Eval[scala.Double], Eval[scala.Double]] =:= ToAst.OfType[Int, Double#Batch, Double]]
  }
}
