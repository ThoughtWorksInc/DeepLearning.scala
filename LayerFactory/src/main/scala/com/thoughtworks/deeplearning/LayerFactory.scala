package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Tape
import scala.language.experimental.macros
import scala.reflect.macros.whitebox

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object LayerFactory {

  private[deeplearning] final class Macros(val c: whitebox.Context) {
    import c.universe._
    def layer(body: Tree): Tree = ???
  }

  trait Hoise {
    def apply[A](a: A): A
  }

  def layer[Input, Output <: Tape](body: (Hoise, Input) => Output): Layer.Aux[Input, Output] = macro Macros.layer

}
