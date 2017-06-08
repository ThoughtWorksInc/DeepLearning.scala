package com.thoughtworks.deeplearning

import java.util.logging.Logger

import com.thoughtworks.feature.{Factory, ImplicitApply}
import Factory.inject

import scala.annotation.meta.getter

trait Hyperparameter {
  type Implicits

  @(inject @getter)
  protected val implicitsFactory: Factory[Implicits]

  @(inject @getter)
  protected val implicitApplyImplicitsConstructor: ImplicitApply[implicitsFactory.Constructor]

  @inject
  protected def isImplicits: implicitApplyImplicitsConstructor.Out <:< Implicits

  @transient
  lazy val implicits: Implicits = {
    isImplicits(implicitApplyImplicitsConstructor(implicitsFactory.newInstance))
  }

  implicit protected def logger: Logger

}
