package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.any.ToNeuralNetwork
import com.thoughtworks.deepLearning.boolean.ast.If

import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object boolean {

  /** @template */
  type Boolean = utilities.Boolean

  final class BooleanOps[Input <: Batch](boolean: NeuralNetwork.Aux[Input, Boolean#Batch]) {

    def `if`[ThatInput <: Input, Output <: Batch](`then`: NeuralNetwork.Aux[ThatInput, Output])(
        `else`: NeuralNetwork.Aux[ThatInput, Output]): NeuralNetwork.Aux[ThatInput, Output] = {
      If[ThatInput, Output](boolean, `then`, `else`)
    }

  }

  implicit def toBooleanOps[From, Input <: Batch](from: From)(
      implicit toNeuralNetwork: ToNeuralNetwork.OfType[From, Input, Boolean]): BooleanOps[Input] = {
    new BooleanOps[Input](toNeuralNetwork(from))
  }

}
