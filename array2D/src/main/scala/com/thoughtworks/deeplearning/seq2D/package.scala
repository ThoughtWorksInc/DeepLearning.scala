package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.any.ToLayer
import com.thoughtworks.deeplearning.seq2D.layers.Get
import com.thoughtworks.deeplearning.double.utilities.Double

import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object seq2D {

  /** @template */
  type Seq2D = utilities.Seq2D

  final class Seq2DOps[Input <: Batch](differentiable: Layer.Aux[Input, Seq2D#Batch]) {
    def apply(rowIndex: Int, columnIndex: Int): Layer.Aux[Input, Double#Batch] =
      Get(differentiable, rowIndex, columnIndex)
  }

  implicit def toSeq2DOps[A, Input <: Batch, OutputData, OutputDelta](a: A)(
      implicit toLayer: ToLayer.OfType[A, Input, Seq2D]): Seq2DOps[Input] = {
    new Seq2DOps(toLayer(a))
  }

}
