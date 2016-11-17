package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.any.ToNeuralNetwork
import com.thoughtworks.deepLearning.boolean.ast.If
import shapeless.Lub

import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object boolean {

  /** @template */
  type Boolean = utilities.Boolean

  final class BooleanOps[Input <: Batch](boolean: NeuralNetwork.Aux[Input, Boolean#Batch]) {

    def `if`[Then,
             Else,
             ThenOutputData,
             ThenOutputDelta,
             ElseOutputData,
             ElseOutputDelta,
             NN,
             OutputData,
             OutputDelta](`then`: Then)(`else`: Else)(
        implicit thenToNeuralNetwork: ToNeuralNetwork.Aux[Then, Input, ThenOutputData, ThenOutputDelta],
        elseToNeuralNetwork: ToNeuralNetwork.Aux[Else, Input, ElseOutputData, ElseOutputDelta],
        lub: Lub[NeuralNetwork.Aux[Input, Batch.Aux[ThenOutputData, ThenOutputDelta]],
                 NeuralNetwork.Aux[Input, Batch.Aux[ElseOutputData, ElseOutputDelta]],
                 NN],
        commonToNeuralNetwork: ToNeuralNetwork.Aux[NN, Input, OutputData, OutputDelta]
    ): NeuralNetwork.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = {
      If[Input, OutputData, OutputDelta](boolean,
                                         commonToNeuralNetwork(lub.left(thenToNeuralNetwork(`then`))),
                                         commonToNeuralNetwork(lub.right(elseToNeuralNetwork(`else`))))
    }

  }

  implicit def toBooleanOps[From, Input <: Batch](from: From)(
      implicit toNeuralNetwork: ToNeuralNetwork.OfType[From, Input, Boolean]): BooleanOps[Input] = {
    new BooleanOps[Input](toNeuralNetwork(from))
  }

}
