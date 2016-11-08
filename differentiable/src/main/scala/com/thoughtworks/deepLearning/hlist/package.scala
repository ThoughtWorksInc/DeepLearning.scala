package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.DifferentiableFunction.{IsAst, Ast}
import com.thoughtworks.deepLearning.Differentiable.Batch
import hlist.ast._
import any._
import com.thoughtworks.deepLearning.any.ast.Identity

import scala.language.implicitConversions
import scalaz.Liskov.<~<

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object hlist {

  implicit final class HListOps[TailAst](val tail: TailAst) {

    def ::[Input0 <: Differentiable,
           HeadAst,
           HeadData,
           HeadDelta,
           TailData <: shapeless.HList,
           TailDelta <: shapeless.Coproduct](head: HeadAst)(
        implicit unapplyHead: IsAst[HeadAst, Input0, HeadData, HeadDelta],
        unapplyTail: IsAst[TailAst, Input0, TailData, TailDelta]
    ): DifferentiableFunction.Ast[Input0, Differentiable.Batch[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]] = {
      HCons[Input0, HeadData, HeadDelta, TailData, TailDelta](unapplyHead(head), unapplyTail(tail))
    }

  }

  implicit final class HConsOps[Input <: Differentiable, HeadData, HeadDelta, TailData <: shapeless.HList,
  TailDelta <: shapeless.Coproduct](
      val hcons: DifferentiableFunction.Ast[Input, Differentiable.Batch[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]) {
    def head = Head(hcons)

    def tail = Tail(hcons)
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

  def hnil[Input <: Differentiable: Identity]: DifferentiableFunction.Ast[Input, Differentiable.Batch[shapeless.HNil, shapeless.CNil]] = HNil

  /** @template */
  type ::[Head <: Differentiable, Tail <: HList] = HList {
    type Data = shapeless.::[Head#Data, Tail#Data]
    type Delta = shapeless.:+:[Head#Delta, Tail#Delta]
  }
}
