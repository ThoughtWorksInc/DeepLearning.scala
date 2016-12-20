package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.Conversion._
import com.thoughtworks.deeplearning.seq.layers.{Get, ToSeq}

import language.implicitConversions

// TODO: rename to sized

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object seq {

  type BpSeq[A <: BackPropagationType[_, _]] =
    BackPropagationType[Seq[BackPropagationType.DataOf[A]], (Int, BackPropagationType.DeltaOf[A])]

  object BpSeq {
    def apply[From, Input <: Batch](layers: From*)(
      implicit elementToLayer: ToLayer[From, Input]
    ): Layer.Aux[Input, Batch.Aux[Seq[elementToLayer.OutputData], (Int, elementToLayer.OutputDelta)]] = {
      ToSeq[Input, elementToLayer.OutputData, elementToLayer.OutputDelta](layers.map(elementToLayer(_)))
    }
  }

  final class SeqLayerOps[Input <: Batch, ElementData, ElementDelta](
      seqLayer: Layer.Aux[Input, Batch.Aux[Seq[ElementData], (Int, ElementDelta)]]) {

    def apply(i: Int): Layer.Aux[Input, Batch.Aux[ElementData, ElementDelta]] = {
      Get[Input, ElementData, ElementDelta](seqLayer, i)
    }

  }

  implicit def toSeqLayerOps[From, Input <: Batch, SeqData, SeqDelta, ElementData, ElementDelta](from: From)(
      implicit toLayer: ToLayer.Aux[From, Input, SeqData, SeqDelta],
      toSeqLayer: Layer.Aux[Input, Batch.Aux[SeqData, SeqDelta]] <:< Layer.Aux[
        Input,
        Batch.Aux[Seq[ElementData], (Int, ElementDelta)]]
  ): SeqLayerOps[Input, ElementData, ElementDelta] = {
    new SeqLayerOps[Input, ElementData, ElementDelta](toSeqLayer(toLayer(from)))
  }

  implicit def seqToLayer[Input0 <: Batch, ElementData, ElementDelta]
    : ToLayer.Aux[Layer.Aux[
                    Input0,
                    Batch.Aux[Seq[ElementData], (Int, ElementDelta)]
                  ],
                  Input0,
                  Seq[ElementData],
                  (Int, ElementDelta)] = {
    new ToLayer[com.thoughtworks.deeplearning.Layer.Aux[
                  Input0,
                  Batch.Aux[Seq[ElementData], (Int, ElementDelta)]
                ],
                Input0] {
      type OutputData = Seq[ElementData]
      type OutputDelta = (Int, ElementDelta)

      override def apply(
          layer: Layer.Aux[
            Input0,
            Batch.Aux[Seq[ElementData], (Int, ElementDelta)]
          ]) = layer
    }
  }

}
