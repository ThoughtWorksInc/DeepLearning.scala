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

//
//  /** @template */
//  type InputAst[InputTypePair <: Any] = Identity[InputTypePair#ConcreteBatch]
//
//  implicit def input[Input <: Batch] = {
//    Identity[Input]()
//  }
//
//  def `throw`(throwable: => Throwable) = {
//    Throw(throwable _)
//  }
//
//  implicit final class NativeAnyOps[Data](data: Data) {
//
//    def toLiteral[Input <: Batch: Identity]: NeuralNetwork.Aux[Input, Batch.ConcreteBatch[Data, scala.Any]] = Literal[Data](data)
//    def toBatch: Batch.ConcreteBatch[Data, scala.Any] = Literal[Data](data)
//
//  }
//
  final class AnyOps[Input <: Batch, OutputData, OutputDelta, NewInputData, NewInputDelta](
      val f: NeuralNetwork.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) {

    def compose(g: NeuralNetwork.Aux[Batch.Aux[NewInputData, NewInputDelta], Input])
      : NeuralNetwork.Aux[Batch.Aux[NewInputData, NewInputDelta], Batch.Aux[OutputData, OutputDelta]] = {
      Compose(f, g)
    }

  }

  implicit def toAnyOps[F, NewInputData, NewInputDelta, Input <: Batch, OutputData, OutputDelta](f: F)(
    implicit toNeuralNetwork: ToNeuralNetwork.Aux[F, Input, OutputData, OutputDelta],
    differentiableType: Type[NewInputData, NewInputDelta])
    : AnyOps[Input, OutputData, OutputDelta, NewInputData, NewInputDelta] = new AnyOps(toNeuralNetwork(f))
}
