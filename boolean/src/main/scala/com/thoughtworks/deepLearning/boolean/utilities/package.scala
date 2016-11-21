package com.thoughtworks.deepLearning.boolean

import cats.Eval
import com.thoughtworks.deepLearning.any._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object utilities {

  type Boolean = Type[Eval[scala.Boolean], Eval[scala.Boolean]]
}
