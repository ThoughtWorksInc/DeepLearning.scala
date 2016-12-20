package com.thoughtworks.deeplearning.double

import cats._
import com.thoughtworks.deeplearning.Conversion._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object utilities {
  type BpDouble = BackPropagationType[Eval[Double], Eval[Double]]
}
