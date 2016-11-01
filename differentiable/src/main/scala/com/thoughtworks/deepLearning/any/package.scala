package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.any.ast.{Identity, Literal, Throw}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object any {

  type Any = {
    type Data
    type Delta
  }

  type InputAst[InputTypePair <: Any] = Identity[Batch.FromTypePair[InputTypePair]]

  implicit def input[Input <: Batch] = {
    Identity[Input]()
  }

  def `throw`(throwable: => Throwable) = {
    Throw(Eval.later(throwable))
  }

  implicit final class NativeAnyOps[Data](data: Data) {

    def toLiteral[Input <: Batch: Identity]: Ast.Aux[Input, Batch.Aux[Data, scala.Any]] = Literal[Data](data)
    def toBatch: Batch.Aux[Data, scala.Any] = Literal[Data](data)

  }

}
