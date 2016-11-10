package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.DifferentiableType.ConcreteType
//import com.thoughtworks.deepLearning.DifferentiableFunction._
//import com.thoughtworks.deepLearning.Differentiable._
//import com.thoughtworks.deepLearning.any.Any
//import com.thoughtworks.deepLearning.boolean.ast.If
//import com.thoughtworks.deepLearning.boolean.utilities._
//import com.thoughtworks.deepLearning.coproduct.ast.{Head, IsInl, Tail}
//import shapeless.Lub
import scala.language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object coproduct {

  /** @template */
  type Coproduct = DifferentiableType {
    type Data <: shapeless.Coproduct
    type Delta <: shapeless.Coproduct
  }

  /** @template */
  type CNil = ConcreteType[shapeless.CNil, shapeless.CNil]

  /** @template */
  type :+:[Head <: DifferentiableType, Tail <: Coproduct] =
    ConcreteType[shapeless.:+:[head.Data, tail.Data], shapeless.:+:[head.Delta, tail.Delta]] forSome {
      val head: Head
      val tail: Tail
    }
//
//  implicit final class CConsOps[
//      Input <: Differentiable,
//      HeadData,
//      HeadDelta,
//      TailData <: shapeless.Coproduct,
//      TailDelta <: shapeless.Coproduct
//  ](
//      ccons: DifferentiableFunction.Ast[
//        Input,
//        Differentiable.Batch[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]
//      ]
//  ) {
//
//    def head: Ast[Input, Batch[HeadData, HeadDelta]] =
//      Head[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)
//
//    def tail: Ast[Input, Batch[TailData, TailDelta]] =
//      Tail[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)
//
//    def choice[HeadCase, TailCase, Output <: Differentiable](
//        caseHead: Ast[Input, Batch[HeadData, HeadDelta]] => Ast[Input, Output])(
//        caseTail: Ast[Input, Batch[TailData, TailDelta]] => Ast[Input, Output])
//      : DifferentiableFunction.Ast[Input, Output] = {
//      If[Input, Output](isInl, caseHead(head), caseTail(tail))
//    }
//
//    def isInl: Ast[Input, Boolean#Batch] = IsInl[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)
//
//  }
//
}
