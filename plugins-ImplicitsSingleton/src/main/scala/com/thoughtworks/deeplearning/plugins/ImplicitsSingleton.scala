package com.thoughtworks.deeplearning
package plugins
import java.util.logging.Logger

import com.thoughtworks.feature.{Factory, ImplicitApply}, ImplicitApply.ops._
import Factory.inject

import scala.annotation.meta.getter

/** A plugin that creates the instance of [[implicits]].
  *
  * Any fields and methods in [[Implicits]] added by other plugins will be mixed-in and present in [[implicits]].
  */
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
