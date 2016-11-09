package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.Differentiable._
import com.thoughtworks.deepLearning.any.Any
import com.thoughtworks.deepLearning.boolean.ast.If
import com.thoughtworks.deepLearning.coproduct.ast.{Head, IsInl, Tail}
import shapeless.Lub

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
      differentiable: DifferentiableFunction.Ast[
        Input,
        Differentiable.Batch[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]) {

    def head: Ast[Input, Batch[HeadData, HeadDelta]] =
      Head[Input, HeadData, HeadDelta, TailData, TailDelta](differentiable)

    def tail: Ast[Input, Batch[TailData, TailDelta]] =
      Tail[Input, HeadData, HeadDelta, TailData, TailDelta](differentiable)

    def choice[NewInput <: Input, HeadCase, TailCase, Output <: Differentiable](
        caseHead: Ast[Input, Batch[HeadData, HeadDelta]] => Ast[NewInput, Output])(
        caseTail: Ast[Input, Batch[TailData, TailDelta]] => Ast[NewInput, Output])
      : DifferentiableFunction.Ast[NewInput, Output] = {
      If[NewInput, Output](isInl, caseHead(head), caseTail(tail))
    }

    def isInl = IsInl[Input, HeadData, HeadDelta, TailData, TailDelta](differentiable)

  }

}
