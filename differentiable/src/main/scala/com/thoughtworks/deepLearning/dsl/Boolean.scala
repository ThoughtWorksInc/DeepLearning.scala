package com.thoughtworks.deepLearning.dsl

import cats.Eval

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait Boolean extends Any {
  override type Delta = Eval[scala.Boolean]
  override type Data = Eval[scala.Boolean]

}
