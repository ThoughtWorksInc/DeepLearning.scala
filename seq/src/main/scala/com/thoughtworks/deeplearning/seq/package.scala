package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.dsl.{ToLayer, Type}
import com.thoughtworks.deeplearning.seq.layers.Get
import scala.language.implicitConversions

// TODO: rename to sized

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object seq {

  type Seq[A <: Type[_, _]] = Type[scala.Seq[Type.DataOf[A]], (Int, Type.DeltaOf[A])]

  final class SeqLayerOps[Input <: Batch, ElementData, ElementDelta](
      seqLayer: Layer.Aux[Input, Batch.Aux[scala.Seq[ElementData], (Int, ElementDelta)]]) {

    def apply(i: Int): Layer.Aux[Input, Batch.Aux[ElementData, ElementDelta]] = {
      Get[Input, ElementData, ElementDelta](seqLayer, i)
    }

  }

  implicit def toSeqLayerOps[From, Input <: Batch, SeqData, SeqDelta, ElementData, ElementDelta](from: From)(
      implicit toLayer: ToLayer.Aux[From, Input, SeqData, SeqDelta],
      toSeqLayer: Layer.Aux[Input, Batch.Aux[SeqData, SeqDelta]] <:< Layer.Aux[
        Input,
        Batch.Aux[scala.Seq[ElementData], (Int, ElementDelta)]]
  ): SeqLayerOps[Input, ElementData, ElementDelta] = {
    new SeqLayerOps[Input, ElementData, ElementDelta](toSeqLayer(toLayer(from)))
  }

}
