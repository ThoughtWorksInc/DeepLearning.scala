package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.Ast.{IsAst, WidenAst}
import com.thoughtworks.deepLearning.Batch.WidenBatch
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

    def ::[Input0 <: Batch,
           HeadAst,
           HeadData,
           HeadDelta,
           TailData <: shapeless.HList,
           TailDelta <: shapeless.Coproduct](head: HeadAst)(
        implicit unapplyHead: IsAst[HeadAst, Input0, HeadData, HeadDelta],
        unapplyTail: IsAst[TailAst, Input0, TailData, TailDelta]
    ) = {
      HCons[Input0, HeadData, HeadDelta, TailData, TailDelta](unapplyHead(head), unapplyTail(tail))
    }

  }

  //  implicit def hlistOpsUnapply[Tail,
  //                               Input0 <: Batch,
  //                               OutputData <: shapeless.HList,
  //                               OutputDelta <: shapeless.Coproduct](tail: Tail)(
  //      implicit unapplyTail: UnapplyOutput.Aux[Tail, Input0, OutputData, OutputDelta])
  //    : HListOps[Input0, OutputData, OutputDelta] =
  //    new HListOps[Input0, OutputData, OutputDelta](unapplyTail(tail))

  implicit final class HConsOps[Input <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList,
  TailDelta <: shapeless.Coproduct](
      val hcons: WidenAst[Input, WidenBatch[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]) {
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

  def hnil[Input <: Batch: Identity]: WidenAst[Input, WidenBatch[shapeless.HNil, shapeless.CNil]] = HNil

  /** @template */
  type ::[Head <: Batch, Tail <: HList] = HList {
    type Data = shapeless.::[Head#Data, Tail#Data]
    type Delta = shapeless.:+:[Head#Delta, Tail#Delta]
  }
}
