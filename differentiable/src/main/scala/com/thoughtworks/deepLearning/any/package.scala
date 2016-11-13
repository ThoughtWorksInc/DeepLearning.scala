package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.Differentiable._
import com.thoughtworks.deepLearning.any.ast.{Compose, Identity, Literal, Throw}
import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object any {

  /** @template */
  type Any = DifferentiableType[_, _]

//
//  /** @template */
//  type InputAst[InputTypePair <: Any] = Identity[InputTypePair#ConcreteBatch]
//
//  implicit def input[Input <: Differentiable] = {
//    Identity[Input]()
//  }
//
//  def `throw`(throwable: => Throwable) = {
//    Throw(throwable _)
//  }
//
//  implicit final class NativeAnyOps[Data](data: Data) {
//
//    def toLiteral[Input <: Differentiable: Identity]: DifferentiableFunction.Ast[Input, Differentiable.ConcreteBatch[Data, scala.Any]] = Literal[Data](data)
//    def toBatch: Differentiable.ConcreteBatch[Data, scala.Any] = Literal[Data](data)
//
//  }
//
  final class AnyOps[Input <: Differentiable, OutputData, OutputDelta, NewInputData, NewInputDelta](
      val f: Ast[Input, Batch[OutputData, OutputDelta]]) {

    def compose(g: Ast[Batch[NewInputData, NewInputDelta], Input])
      : Ast[Batch[NewInputData, NewInputDelta], Batch[OutputData, OutputDelta]] = {
      Compose(f, g)
    }

  }

  implicit def toAnyOps[F, NewInputData, NewInputDelta, Input <: Differentiable, OutputData, OutputDelta](f: F)(
      implicit toAst: ToAst.Aux[F, Input, OutputData, OutputDelta],
      differentiableType: DifferentiableType[NewInputData, NewInputDelta])
    : AnyOps[Input, OutputData, OutputDelta, NewInputData, NewInputDelta] = new AnyOps(toAst(f))
}
