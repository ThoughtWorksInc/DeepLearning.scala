package com.thoughtworks.deeplearning

import cats.Eval
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.DifferentiableInt._
import com.thoughtworks.deeplearning.DifferentiableSeq._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.DifferentiableDouble.Optimizers.LearningRate
import org.scalatest._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.Poly.MathOps
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import shapeless._

import language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class SeqSpec extends FreeSpec with Matchers {

  implicit def learningRate: LearningRate = new LearningRate {
    override protected def currentLearningRate() = 0.03
  }

  def unsafe(implicit s: SeqPlaceholder[AnyPlaceholder]) = {
    s(0).asInstanceOf[s.To[DoublePlaceholder]] - 1.0.toWeight
  }

  "erased SeqPlaceholder" in {
    val unsafeNetwork = unsafe
    unsafeNetwork.train(Seq(2.4))
  }

  //TODO:compile error in scala version 2.12 --issues #11
  /*"Seq(Int).toLayer" in {
    //noinspection ScalaUnusedSymbol
    def toLayerTest(implicit from: From[Double]##`@`) = {
      1.0.toLayer
      Seq(1.0).toLayer
      Seq(1.0.toLayer).toLayer
      Seq(1.toLayer).toLayer
      Seq(1).toLayer
    }
  }*/
}
