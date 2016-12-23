package com.thoughtworks.deeplearning

import cats.Eval
import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.DifferentiableSeq._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.DifferentiableDouble.Optimizers.LearningRate
import org.scalatest._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.Poly.MathOps
import com.thoughtworks.deeplearning.Poly.MathFunctions._

import language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class SeqSpec extends FreeSpec with Matchers {

  implicit def learningRate: LearningRate = new LearningRate {
    override protected def currentLearningRate() = 0.03
  }

  def unsafe(implicit s: DifferentiableSeq[AnyPlaceholder]) = {
    s(0).asInstanceOf[s.To[DoublePlaceholder]] - 1.0.toWeight
  }

  "erased DifferentiableSeq" in {
    val unsafeNetwork = unsafe
    unsafeNetwork.train(Seq(2.4))
  }
}
