package com.thoughtworks.deeplearning.seq2D

import cats.Eval
import com.thoughtworks.deeplearning.Batch
import com.thoughtworks.deeplearning.dsl.Type

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object utilities {

  /**
    * TODO: allow type parameters
    */
  private[deeplearning] type Seq2D =
    Type[Eval[Seq[Seq[scala.Double]]], Eval[(scala.Int, scala.Int, scala.Double)]]

}
