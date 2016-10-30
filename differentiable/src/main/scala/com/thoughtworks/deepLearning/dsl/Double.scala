package com.thoughtworks.deepLearning.dsl

import cats.Eval

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
sealed trait Double extends Any {
  override type Delta = Eval[scala.Double]
  override type Data = Eval[scala.Double]

}
