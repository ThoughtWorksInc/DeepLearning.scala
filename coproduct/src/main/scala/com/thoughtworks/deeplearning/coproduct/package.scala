package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.boolean.utilities._

import com.thoughtworks.deeplearning.any.{ToLayer, Type}
import com.thoughtworks.deeplearning.any.Type.{DataOf, DeltaOf}
import com.thoughtworks.deeplearning.boolean.layers.If
import com.thoughtworks.deeplearning.coproduct.layers._
import shapeless.Lub

import scala.language.existentials
import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object coproduct {

  /** @template */
  type Coproduct = Type[_ <: shapeless.Coproduct, _ <: shapeless.Coproduct]

  /** @template */
  type CNil = Type[shapeless.CNil, shapeless.CNil]

  /** @template */
  type :+:[Head <: Type[_, _], Tail <: Coproduct] =
    Type[shapeless.:+:[DataOf[Head], DataOf[Tail]], shapeless.:+:[DeltaOf[Head], DeltaOf[Tail]]]

  final class CConsOps[
      Input <: Batch,
      HeadData,
      HeadDelta,
      TailData <: shapeless.Coproduct,
      TailDelta <: shapeless.Coproduct
  ](
      ccons: Layer.Aux[
        Input,
        Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]
      ]
  ) {

    def head: Layer.Aux[Input, Batch.Aux[HeadData, HeadDelta]] =
      Head[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)

    def tail: Layer.Aux[Input, Batch.Aux[TailData, TailDelta]] =
      Tail[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)

    def choice[HeadCase,
               TailCase,
               HeadOutputData,
               HeadOutputDelta,
               TailOutputData,
               TailOutputDelta,
               NN,
               OutputData,
               OutputDelta](caseHead: Layer.Aux[Input, Batch.Aux[HeadData, HeadDelta]] => HeadCase)(
        caseTail: Layer.Aux[Input, Batch.Aux[TailData, TailDelta]] => TailCase)(
        implicit headToLayer: ToLayer.Aux[HeadCase, Input, HeadOutputData, HeadOutputDelta],
        tailToLayer: ToLayer.Aux[TailCase, Input, TailOutputData, TailOutputDelta],
        lub: Lub[Layer.Aux[Input, Batch.Aux[HeadOutputData, HeadOutputDelta]],
                 Layer.Aux[Input, Batch.Aux[TailOutputData, TailOutputDelta]],
                 NN],
        commonToLayer: ToLayer.Aux[NN, Input, OutputData, OutputDelta]
    ): Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = {
      If[Input, OutputData, OutputDelta](isInl,
                                         commonToLayer(lub.left(headToLayer(caseHead(head)))),
                                         commonToLayer(lub.right(tailToLayer(caseTail(tail)))))
    }

    def isInl: Layer.Aux[Input, Boolean#Batch] = IsInl[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)

  }

  implicit def toCConsOps[From,
                          Input <: Batch,
                          OutputData,
                          OutputDelta,
                          HeadData,
                          HeadDelta,
                          TailData <: shapeless.Coproduct,
                          TailDelta <: shapeless.Coproduct](from: From)(
      implicit toLayer: ToLayer.Aux[From, Input, OutputData, OutputDelta],
      toCoproductLayer: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] <:< Layer.Aux[
        Input,
        Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
  ): CConsOps[Input, HeadData, HeadDelta, TailData, TailDelta] = {
    new CConsOps[Input, HeadData, HeadDelta, TailData, TailDelta](toCoproductLayer(toLayer(from)))
  }
}
