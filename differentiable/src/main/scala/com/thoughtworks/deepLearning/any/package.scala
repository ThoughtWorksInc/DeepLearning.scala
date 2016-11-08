package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.Differentiable._
import cats.Eval
import com.thoughtworks.deepLearning.any.ast.{Compose, Identity, Literal, Throw}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object any {

  /** @template */
  type Any = Differentiable

  /** @template */
  type InputAst[InputTypePair <: Any] = Identity[InputTypePair#Widen]

  implicit def input[Input <: Differentiable] = {
    Identity[Input]()
  }

  def `throw`(throwable: => Throwable) = {
    Throw(throwable _)
  }

  implicit final class NativeAnyOps[Data](data: Data) {

    def toLiteral[Input <: Differentiable: Identity]: Ast[Input, Batch[Data, scala.Any]] = Literal[Data](data)
    def toBatch: Batch[Data, scala.Any] = Literal[Data](data)

  }

  implicit final class AnyOps[Input <: Differentiable, Output <: Differentiable](f: Ast[Input, Output]) {

    def compose[NewInput <: Differentiable](g: Ast[NewInput, Input]): Ast[NewInput, Output] = {
      Compose[NewInput, Input, Output](f, g)
    }

  }

}
