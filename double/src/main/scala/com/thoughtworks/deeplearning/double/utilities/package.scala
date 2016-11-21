package com.thoughtworks.deeplearning.double

import cats._
import com.thoughtworks.deeplearning.any.Type

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object utilities {
  type Double = Type[Eval[scala.Double], Eval[scala.Double]]
}
