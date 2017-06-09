package com.thoughtworks.deeplearning
package plugins
import java.util.logging.Logger

import com.thoughtworks.feature.{Factory, ImplicitApply}, ImplicitApply.ops._
import Factory.inject

import scala.annotation.meta.getter

trait ImplicitsSingleton {
  type Implicits

  @inject
  protected val implicitsFactory: Factory[Implicits]

  @inject
  protected val implicitApplyImplicitsConstructor: ImplicitApply[implicitsFactory.Constructor]

  @inject
  protected def asImplicits: implicitApplyImplicitsConstructor.Out <:< Implicits

  @transient
  lazy val implicits: Implicits = {
    asImplicits(implicitApplyImplicitsConstructor(implicitsFactory.newInstance))
  }

}
