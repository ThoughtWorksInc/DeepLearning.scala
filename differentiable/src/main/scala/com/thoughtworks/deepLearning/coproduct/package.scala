package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.Ast._
import com.thoughtworks.deepLearning.Batch._
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

  implicit final class CConsOps[Input <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
  TailDelta <: shapeless.Coproduct](
      differentiable: WidenAst[Input,
                               WidenBatch[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]) {

    def head = Head[Input, HeadData, HeadDelta, TailData, TailDelta](differentiable)

    def tail = Tail[Input, HeadData, HeadDelta, TailData, TailDelta](differentiable)

    def choice[ThatInput <: Input, Output <: Batch](
        caseHead: WidenAst[Input, WidenBatch[HeadData, HeadDelta]] => WidenAst[ThatInput, Output])(
        caseTail: WidenAst[Input, WidenBatch[TailData, TailDelta]] => WidenAst[ThatInput, Output])
      : WidenAst[ThatInput, Output] = {
      If[ThatInput, Output](isInl, caseHead(head), caseTail(tail))
    }

    def isInl = IsInl[Input, HeadData, HeadDelta, TailData, TailDelta](differentiable)

  }

}
