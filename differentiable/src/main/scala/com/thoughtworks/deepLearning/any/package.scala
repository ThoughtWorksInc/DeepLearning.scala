package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.Ast._
import com.thoughtworks.deepLearning.Batch._
import cats.Eval
import com.thoughtworks.deepLearning.any.ast.{Identity, Literal, Throw}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object any {

  /** @template */
  type Any = Batch

  /** @template */
  type InputAst[InputTypePair <: Any] = Identity[InputTypePair#Widen]

  implicit def input[Input <: Batch] = {
    Identity[Input]()
  }

  def `throw`(throwable: => Throwable) = {
    Throw(Eval.later(throwable))
  }

  implicit final class NativeAnyOps[Data](data: Data) {

    def toLiteral[Input <: Batch: Identity]: WidenAst[Input, WidenBatch[Data, scala.Any]] = Literal[Data](data)
    def toBatch: WidenBatch[Data, scala.Any] = Literal[Data](data)

  }

}
