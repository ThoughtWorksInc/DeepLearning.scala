package com.thoughtworks.deeplearning

import scala.language.experimental.macros
import scala.reflect.macros.whitebox

final case class Caller[A](value: A)
object Caller {
  implicit def generate: Caller[_] = macro impl

  def impl(c: whitebox.Context): c.Tree = {
    import c.universe._
    q"new _root_.com.thoughtworks.deeplearning.Caller(this)"
  }
}
