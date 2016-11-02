package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.Ast.WidenAst
import com.thoughtworks.deepLearning.Batch.WidenBatch
import hlist.ast._
import any._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object hlist {

  implicit final class HConsOps[Input <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList,
  TailDelta <: shapeless.Coproduct](
      val differentiable: WidenAst[
        Input,
        WidenBatch[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]) {
    def head = Head(differentiable)

    def tail = Tail(differentiable)
  }

  /** @template */
  type HList = Any {
    type Data <: shapeless.HList
    type Delta <: shapeless.Coproduct
  }

  /** @template */
  type HNil = HList {
    type Data = shapeless.HNil
    type Delta = shapeless.CNil
  }

  /** @template */
  type ::[Head <: Batch, Tail <: HList] = HList {
    type Data = shapeless.::[Head#Data, Tail#Data]
    type Delta = shapeless.:+:[Head#Delta, Tail#Delta]
  }
}
