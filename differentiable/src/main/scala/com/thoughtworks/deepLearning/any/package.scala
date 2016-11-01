package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.any.ast.{Identity, Throw}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object any {

  type Any = {
    type Data
    type Delta
  }

  def input[Input <: Batch] = {
    Identity[Input]()
  }

  def `throw`(throwable: => Throwable) = {
    Throw(Eval.later(throwable))
  }

}
