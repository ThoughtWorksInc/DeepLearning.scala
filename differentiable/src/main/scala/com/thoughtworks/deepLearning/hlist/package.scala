package com.thoughtworks.deepLearning
//
//import com.thoughtworks.deepLearning.DifferentiableFunction.{Ast, ToAst}
//import hlist.ast._
//import any._
//import com.thoughtworks.deepLearning.Differentiable.ConcreteBatch
//import com.thoughtworks.deepLearning.any.ast.Identity
//
import com.thoughtworks.deepLearning.Differentiable.Batch
import com.thoughtworks.deepLearning.DifferentiableFunction.Ast
import com.thoughtworks.deepLearning.hlist.ast._

import scala.language.implicitConversions
import scala.language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object hlist {

  /** @template */
  type HList = DifferentiableType[_ <: shapeless.HList, _ <: shapeless.Coproduct]

  /** @template */
  type HNil = DifferentiableType[shapeless.HNil, shapeless.CNil]

  /** @template */
  type ::[Head <: DifferentiableType[_, _], Tail <: HList] =
    DifferentiableType[shapeless.::[head.Data, tail.Data], shapeless.:+:[head.Delta, tail.Delta]] forSome {
      val head: Head
      val tail: Tail
    }
//
//  implicit final class HListOps[TailAst](val tail: TailAst) {
//
//    def ::[Input0 <: Differentiable,
//           HeadAst,
//           HeadData,
//           HeadDelta,
//           TailData <: shapeless.HList,
//           TailDelta <: shapeless.Coproduct](head: HeadAst)(
//        implicit unapplyHead: ToAst[HeadAst, Input0, HeadData, HeadDelta],
//        unapplyTail: ToAst[TailAst, Input0, TailData, TailDelta]
//    ): Ast[Input0, ConcreteBatch[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]] = {
//      HCons[Input0, HeadData, HeadDelta, TailData, TailDelta](unapplyHead(head), unapplyTail(tail))
//    }
//
//  }
  val HNil = ast.HNil

  implicit final class HConsOps[Input <: Differentiable, HeadData, HeadDelta, TailData <: shapeless.HList,
  TailDelta <: shapeless.Coproduct](
      hcons: Ast[Input, Batch[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]) {
    def head: Ast[Input, Batch[HeadData, HeadDelta]] = Head[Input, HeadData, HeadDelta, TailData, TailDelta](hcons)

    def tail: Ast[Input, Batch[TailData, TailDelta]] = Tail[Input, HeadData, HeadDelta, TailData, TailDelta](hcons)
  }

  implicit def toHConsOps[From,
                          Input <: Differentiable,
                          OutputData,
                          OutputDelta,
                          HeadData,
                          HeadDelta,
                          TailData <: shapeless.HList,
                          TailDelta <: shapeless.Coproduct](from: From)(
      implicit toAst: ToAst.Aux[From, Input, OutputData, OutputDelta],
      toHListAst: Ast[Input, Batch[OutputData, OutputDelta]] <:< Ast[
        Input,
        Batch[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
  ): HConsOps[Input, HeadData, HeadDelta, TailData, TailDelta] = {
    new HConsOps[Input, HeadData, HeadDelta, TailData, TailDelta](toHListAst(toAst(from)))
  }

}
