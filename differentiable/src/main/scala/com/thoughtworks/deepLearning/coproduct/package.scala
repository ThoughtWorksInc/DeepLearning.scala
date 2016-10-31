package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.any.Any
import com.thoughtworks.deepLearning.boolean.ast.If
import com.thoughtworks.deepLearning.coproduct.ast.{Head, Tail, IsInl}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object coproduct {

  type Coproduct = {
    type Data <: shapeless.Coproduct
    type Delta <: shapeless.Coproduct
  }

  type CNil = {
    type Data = shapeless.CNil
    type Delta = shapeless.CNil
  }

  type :+:[Head <: Any, Tail <: Coproduct] = {
    type Data = shapeless.:+:[Head#Data, Tail#Data]
    type Delta = shapeless.:+:[Head#Delta, Tail#Delta]
  }

  implicit final class CConsOps[Input <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
  TailDelta <: shapeless.Coproduct](
      differentiable: Ast.Aux[
        Input,
        Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]) {

    lazy val head = Head(differentiable)

    lazy val tail = Tail(differentiable)

    def choice[ThatInput <: Input, Output <: Batch](caseHead: head.type => Ast.Aux[ThatInput, Output])(
        caseTail: tail.type => Ast.Aux[ThatInput, Output]) = {
      If[ThatInput, Output](IsInl(differentiable), caseHead(head), caseTail(tail))
    }

  }
}
