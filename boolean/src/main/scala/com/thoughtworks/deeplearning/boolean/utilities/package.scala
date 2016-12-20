package com.thoughtworks.deeplearning.boolean

import cats.Eval
import com.thoughtworks.deeplearning.dsl._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object utilities {

  type BpBoolean = BackPropagationType[Eval[scala.Boolean], Eval[scala.Boolean]]
}
