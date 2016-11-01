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
      differentiable: Ast.Aux[Input,
                              Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]) {

    def head = Head(differentiable)

    def tail = Tail(differentiable)

    def choice[ThatInput <: Input, Output <: Batch](
        caseHead: Ast.Aux[Input, Batch.Aux[HeadData, HeadDelta]] => Ast.Aux[ThatInput, Output])(
        caseTail: Ast.Aux[Input, Batch.Aux[TailData, TailDelta]] => Ast.Aux[ThatInput, Output]) = {
      If(isInl, caseHead(head), caseTail(tail))
    }

    def isInl = IsInl(differentiable)

  }

}
