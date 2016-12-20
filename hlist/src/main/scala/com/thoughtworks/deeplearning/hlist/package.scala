package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.dsl.{BackPropagationType, ToLayer}
import com.thoughtworks.deeplearning.dsl.BackPropagationType.{DataOf, DeltaOf}
import com.thoughtworks.deeplearning.hlist.layers._
import shapeless._

import language.implicitConversions
import language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object hlist {

  /** @template */
  type BpHList = BackPropagationType[_ <: HList, _ <: Coproduct]

  /** @template */
  type BpHNil = BackPropagationType[HNil, CNil]

  /** @template */
  type BpHCons[Head <: BackPropagationType[_, _], Tail <: BpHList] =
    BackPropagationType[DataOf[Head] :: DataOf[Tail], DeltaOf[Head] :+: DeltaOf[Tail]]

  /** @template */
  type :**:[Head <: BackPropagationType[_, _], Tail <: BpHList] =
    BackPropagationType[::[DataOf[Head], DataOf[Tail]], :+:[DeltaOf[Head], DeltaOf[Tail]]]

  val BpHNil: layers.HNil.type = layers.HNil

  implicit def hnilToLayer[InputData, InputDelta](implicit inputType: BackPropagationType[InputData, InputDelta])
    : ToLayer.Aux[layers.HNil.type, Batch.Aux[InputData, InputDelta], HNil, CNil] =
    new ToLayer[layers.HNil.type, Batch.Aux[InputData, InputDelta]] {
      override type OutputData = HNil
      override type OutputDelta = CNil

      override def apply(hnil: layers.HNil.type) = hnil
    }

  final class HListOps[Input <: Batch, TailData <: HList, TailDelta <: Coproduct](
      tail: Layer.Aux[Input, Batch.Aux[TailData, TailDelta]]) {

    def ::[Head, HeadData, HeadDelta](head: Head)(implicit headToLayer: ToLayer.Aux[Head, Input, HeadData, HeadDelta])
    : Layer.Aux[Input, Batch.Aux[::[HeadData, TailData], :+:[HeadDelta, TailDelta]]] = {
      HCons[Input, HeadData, HeadDelta, TailData, TailDelta](headToLayer(head), tail)
    }

    def :**:[Head, HeadData, HeadDelta](head: Head)(implicit headToLayer: ToLayer.Aux[Head, Input, HeadData, HeadDelta])
      : Layer.Aux[Input, Batch.Aux[::[HeadData, TailData], :+:[HeadDelta, TailDelta]]] = {
      HCons[Input, HeadData, HeadDelta, TailData, TailDelta](headToLayer(head), tail)
    }

  }

  implicit def toHListOps[From, Input <: Batch, TailData <: HList, TailDelta <: Coproduct](
      from: From)(
      implicit toLayer: ToLayer.Aux[From, Input, TailData, TailDelta]
  ): HListOps[Input, TailData, TailDelta] = {
    new HListOps[Input, TailData, TailDelta](toLayer(from))
  }

  final class HConsOps[Input <: Batch, HeadData, HeadDelta, TailData <: HList,
  TailDelta <: Coproduct](
      hcons: Layer.Aux[Input, Batch.Aux[::[HeadData, TailData], :+:[HeadDelta, TailDelta]]]) {
    def head: Layer.Aux[Input, Batch.Aux[HeadData, HeadDelta]] =
      Head[Input, HeadData, HeadDelta, TailData, TailDelta](hcons)

    def tail: Layer.Aux[Input, Batch.Aux[TailData, TailDelta]] =
      Tail[Input, HeadData, HeadDelta, TailData, TailDelta](hcons)
  }

  implicit def toHConsOps[From,
                          Input <: Batch,
                          OutputData,
                          OutputDelta,
                          HeadData,
                          HeadDelta,
                          TailData <: HList,
                          TailDelta <: Coproduct](from: From)(
      implicit toLayer: ToLayer.Aux[From, Input, OutputData, OutputDelta],
      toHListLayer: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] <:< Layer.Aux[
        Input,
        Batch.Aux[::[HeadData, TailData], :+:[HeadDelta, TailDelta]]]
  ): HConsOps[Input, HeadData, HeadDelta, TailData, TailDelta] = {
    new HConsOps[Input, HeadData, HeadDelta, TailData, TailDelta](toHListLayer(toLayer(from)))
  }

}
