package com.thoughtworks.deepLearning.seq2D

import cats.Eval
import com.thoughtworks.deepLearning.Differentiable

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object utilities {

  /**
    * TODO: allow type parameters
    */
  private[deepLearning] type Seq2D =
    com.thoughtworks.deepLearning.DifferentiableType[Eval[Seq[Seq[scala.Double]]], Eval[(scala.Int, scala.Int, scala.Double)]]

}
