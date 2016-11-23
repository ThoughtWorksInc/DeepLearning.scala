package com.thoughtworks.deeplearning.boolean

import cats.Eval
import com.thoughtworks.deeplearning.any._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object utilities {

  type Boolean = Type[Eval[scala.Boolean], Eval[scala.Boolean]]
}
