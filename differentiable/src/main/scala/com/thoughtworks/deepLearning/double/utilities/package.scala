package com.thoughtworks.deepLearning.double

import cats._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object utilities {
  type Double = com.thoughtworks.deepLearning.DifferentiableType.ConcreteType[Eval[scala.Double], Eval[scala.Double]]
}
