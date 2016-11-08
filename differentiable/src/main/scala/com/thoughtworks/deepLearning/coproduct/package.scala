package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.Differentiable._
import com.thoughtworks.deepLearning.any.Any
import com.thoughtworks.deepLearning.boolean.ast.If
import com.thoughtworks.deepLearning.coproduct.ast.{Head, Tail, IsInl}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object coproduct {

  /** @template */
  type Coproduct = Any {
    type Data <: shapeless.Coproduct
    type Delta <: shapeless.Coproduct
  }

  /** @template */
  type CNil = Coproduct {
    type Data = shapeless.CNil
    type Delta = shapeless.CNil
  }

  /** @template */
  type :+:[Head <: Any, Tail <: Coproduct] = Coproduct {
    type Data = shapeless.:+:[Head#Data, Tail#Data]
    type Delta = shapeless.:+:[Head#Delta, Tail#Delta]
  }

  implicit final class CConsOps[Input <: Differentiable, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
  TailDelta <: shapeless.Coproduct](
      differentiable: Ast[Input,
                               Batch[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]) {

    def head = Head[Input, HeadData, HeadDelta, TailData, TailDelta](differentiable)

    def tail = Tail[Input, HeadData, HeadDelta, TailData, TailDelta](differentiable)

    def choice[ThatInput <: Input, Output <: Differentiable](
        caseHead: Ast[Input, Batch[HeadData, HeadDelta]] => Ast[ThatInput, Output])(
        caseTail: Ast[Input, Batch[TailData, TailDelta]] => Ast[ThatInput, Output])
      : Ast[ThatInput, Output] = {
      If[ThatInput, Output](isInl, caseHead(head), caseTail(tail))
    }

    def isInl = IsInl[Input, HeadData, HeadDelta, TailData, TailDelta](differentiable)

  }

}
