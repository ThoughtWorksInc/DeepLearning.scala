package com.thoughtworks.deepLearning.seq2D

import cats.Eval
import com.thoughtworks.deepLearning.Batch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object utilities {

  /**
    * TODO: allow type parameters
    */
  private[deepLearning] type Seq2D =
    com.thoughtworks.deepLearning.Type[Eval[Seq[Seq[scala.Double]]], Eval[(scala.Int, scala.Int, scala.Double)]]

}
