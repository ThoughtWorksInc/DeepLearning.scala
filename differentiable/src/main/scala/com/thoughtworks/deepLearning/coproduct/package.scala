package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.boolean.utilities._
import com.thoughtworks.deepLearning.Differentiable.Batch
import com.thoughtworks.deepLearning.DifferentiableFunction.Ast
import com.thoughtworks.deepLearning.boolean.ast.If
import com.thoughtworks.deepLearning.coproduct.ast._

import scala.language.existentials
import scala.language.implicitConversions


/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object coproduct {

  /** @template */
  type Coproduct = DifferentiableType[_ <: shapeless.Coproduct, _ <: shapeless.Coproduct]

  /** @template */
  type CNil = DifferentiableType[shapeless.CNil, shapeless.CNil]

  /** @template */
  type :+:[Head <: DifferentiableType[_, _], Tail <: Coproduct] =
    DifferentiableType[shapeless.:+:[head.Data, tail.Data], shapeless.:+:[head.Delta, tail.Delta]] forSome {
      val head: Head
      val tail: Tail
    }

  final class CConsOps[
      Input <: Differentiable,
      HeadData,
      HeadDelta,
      TailData <: shapeless.Coproduct,
      TailDelta <: shapeless.Coproduct
  ](
      ccons: Ast[
        Input,
        Differentiable.Batch[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]
      ]
  ) {

    def head: Ast[Input, DifferentiableType[HeadData, HeadDelta]#Batch] =
      Head[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)

    def tail: Ast[Input, DifferentiableType[TailData, TailDelta]#Batch] =
      Tail[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)

    def choice[HeadCase, TailCase, OutputData, OutputDelta](
        caseHead: Ast[Input, DifferentiableType[HeadData, HeadDelta]#Batch] => HeadCase)(
        caseTail: Ast[Input, DifferentiableType[TailData, TailDelta]#Batch] => TailCase)(
        implicit headToAst: ToAst.Aux[HeadCase, Input, OutputData, OutputDelta],
        tailToAst: ToAst.Aux[TailCase, Input, OutputData, OutputDelta])
      : DifferentiableFunction.Ast[Input, DifferentiableType[OutputData, OutputDelta]#Batch] = {
      If[Input, Batch[OutputData, OutputDelta]](isInl, caseHead(head), caseTail(tail))
    }

    def isInl: Ast[Input, Boolean#Batch] = IsInl[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)

  }

  implicit def toCConsOps[From,
                          Input <: Differentiable,
                          OutputData,
                          OutputDelta,
                          HeadData,
                          HeadDelta,
                          TailData <: shapeless.Coproduct,
                          TailDelta <: shapeless.Coproduct](from: From)(
      implicit toAst: ToAst.Aux[From, Input, OutputData, OutputDelta],
      toCoproductAst: Ast[Input, Batch[OutputData, OutputDelta]] <:< Ast[
        Input,
        Batch[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
  ): CConsOps[Input, HeadData, HeadDelta, TailData, TailDelta] = {
    new CConsOps[Input, HeadData, HeadDelta, TailData, TailDelta](toCoproductAst(toAst(from)))
  }
}
