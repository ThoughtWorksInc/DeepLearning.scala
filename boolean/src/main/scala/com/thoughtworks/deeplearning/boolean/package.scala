package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.dsl.ToLayer
import com.thoughtworks.deeplearning.boolean.layers.If
import shapeless.Lub

import language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object boolean {

  /** @template */
  type BpBoolean = com.thoughtworks.deeplearning.boolean.utilities.BpBoolean

  final class BooleanOps[Input <: Batch](boolean: Layer.Aux[Input, BpBoolean#Batch]) {

    def `if`[Then,
             Else,
             ThenOutputData,
             ThenOutputDelta,
             ElseOutputData,
             ElseOutputDelta,
             NN,
             OutputData,
             OutputDelta](`then`: Then)(`else`: Else)(
        implicit thenToLayer: ToLayer.Aux[Then, Input, ThenOutputData, ThenOutputDelta],
        elseToLayer: ToLayer.Aux[Else, Input, ElseOutputData, ElseOutputDelta],
        lub: Lub[Layer.Aux[Input, Batch.Aux[ThenOutputData, ThenOutputDelta]],
                 Layer.Aux[Input, Batch.Aux[ElseOutputData, ElseOutputDelta]],
                 NN],
        commonToLayer: ToLayer.Aux[NN, Input, OutputData, OutputDelta]
    ): Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = {
      If[Input, OutputData, OutputDelta](boolean,
                                         commonToLayer(lub.left(thenToLayer(`then`))),
                                         commonToLayer(lub.right(elseToLayer(`else`))))
    }

  }

  implicit def toBooleanOps[From, Input <: Batch](from: From)(
      implicit toLayer: ToLayer.OfType[From, Input, BpBoolean]): BooleanOps[Input] = {
    new BooleanOps[Input](toLayer(from))
  }

}
