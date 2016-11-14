package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.NeuralNetwork._
import com.thoughtworks.deepLearning.Batch._
import com.thoughtworks.deepLearning.any.ast.{Compose, Identity, Literal, Throw}
import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object any {

  /** @template */
  type Any = Type[_, _]

  /** @template */
  type Nothing = Type[scala.Nothing, scala.Any]

  def `throw`[InputData, InputDelta](throwable: => Throwable)(implicit inputType: Type[InputData, InputDelta])
    : NeuralNetwork.Aux[Batch.Aux[InputData, InputDelta], Nothing#Batch] = {
    Throw(throwable _)
  }

  implicit def autoToLiteral[A, Input <: Batch, OutputData, OutputDelta](a: A)(
      implicit toNeuralNetwork: ToNeuralNetwork.Aux[A, Input, OutputData, OutputDelta])
    : NeuralNetwork.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = {
    toNeuralNetwork(a)
  }

  final class AnyOps[Input <: Batch, OutputData, OutputDelta](
      val toLiteral: NeuralNetwork.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) {

    def compose[G, NewInput <: Batch, InputData, InputDelta](g: G)(
        implicit differentiableType: ToNeuralNetwork.Aux[G, NewInput, InputData, InputDelta],
        toInput: NeuralNetwork.Aux[NewInput, Batch.Aux[InputData, InputDelta]] <:< NeuralNetwork.Aux[NewInput, Input]
    ): NeuralNetwork.Aux[NewInput, Batch.Aux[OutputData, OutputDelta]] = {
      Compose(toLiteral, toInput(differentiableType(g)))
    }

  }

  implicit def toAnyOps[A, Input <: Batch, OutputData, OutputDelta](a: A)(
      implicit toNeuralNetwork: ToNeuralNetwork.Aux[A, Input, OutputData, OutputDelta])
    : AnyOps[Input, OutputData, OutputDelta] = {
    new AnyOps(toNeuralNetwork(a))
  }

}
