package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.Layer._
import com.thoughtworks.deepLearning.Batch._
import com.thoughtworks.deepLearning.any.ToLayer.{LayerPoly1, LayerPoly2}
import com.thoughtworks.deepLearning.any.layers.{Compose, Identity, Literal, Throw}

import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object any {

  /** @template */
  type Any = Type[_, _]

  /** @template */
  type Nothing = Type[scala.Nothing, scala.Any]

  def `throw`[InputData, InputDelta](throwable: => Throwable)(
      implicit inputType: Type[InputData, InputDelta]): Layer.Aux[Batch.Aux[InputData, InputDelta], Nothing#Batch] = {
    Throw(throwable _)
  }

  implicit def autoToLiteral[A, Input <: Batch, OutputData, OutputDelta](a: A)(
      implicit toLayer: ToLayer.Aux[A, Input, OutputData, OutputDelta])
    : Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = {
    toLayer(a)
  }

  final class AnyOps[Input <: Batch, OutputData, OutputDelta](
      val toLiteral: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) {

    def compose[G, NewInput <: Batch, InputData, InputDelta](g: G)(
        implicit differentiableType: ToLayer.Aux[G, NewInput, InputData, InputDelta],
        toInput: Layer.Aux[NewInput, Batch.Aux[InputData, InputDelta]] <:< Layer.Aux[NewInput, Input]
    ): Layer.Aux[NewInput, Batch.Aux[OutputData, OutputDelta]] = {
      Compose(toLiteral, toInput(differentiableType(g)))
    }

    def predict[InputData, InputDelta](inputData: InputData)(
        implicit ev: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] <:< Layer.Aux[
          Batch.Aux[InputData, InputDelta],
          Batch.Aux[OutputData, OutputDelta]]
    ): OutputData = {
      val outputBatch = toLiteral.forward(Literal[InputData](inputData)).open()
      try {
        outputBatch.value
      } finally {
        outputBatch.close()
      }

    }

    def train[InputData, InputDelta](inputData: InputData)(
        implicit ev: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] <:< Layer.Aux[
          Batch.Aux[InputData, InputDelta],
          Batch.Aux[OutputData, OutputDelta]],
        outputDataIsOutputDelta: OutputData <:< OutputDelta
    ): Unit = {
      val outputBatch = toLiteral.forward(Literal[InputData](inputData)).open()
      try {
        outputBatch.backward(outputDataIsOutputDelta(outputBatch.value))
      } finally {
        outputBatch.close()
      }

    }

  }

  implicit def toAnyOps[A, Input <: Batch, OutputData, OutputDelta](a: A)(
      implicit toLayer: ToLayer.Aux[A, Input, OutputData, OutputDelta]): AnyOps[Input, OutputData, OutputDelta] = {
    new AnyOps(toLayer(a))
  }

  implicit final class ToBatchId[Data](a: Data) {
    def toBatchId[Delta]: BatchId.Aux[Batch.Aux[Data, Delta]] = Literal[Data](a)
  }

  implicit final class ScalaAnyOps[Left](left: Left) {

    def -[Right](right: Right)(implicit methodCase: PolyMethods.-.Case[Left, Right]): methodCase.Result =
      PolyMethods.-(left, right)

    def +[Right](right: Right)(implicit methodCase: PolyMethods.+.Case[Left, Right]): methodCase.Result =
      PolyMethods.+(left, right)

    def *[Right](right: Right)(implicit methodCase: PolyMethods.*.Case[Left, Right]): methodCase.Result =
      PolyMethods.*(left, right)

    def /[Right](right: Right)(implicit methodCase: PolyMethods./.Case[Left, Right]): methodCase.Result =
      PolyMethods./(left, right)

  }

  object log extends LayerPoly1
  object exp extends LayerPoly1
  object abs extends LayerPoly1
  object max extends LayerPoly2
  object min extends LayerPoly2
}
