package com.thoughtworks.deeplearning

import resource._
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.dsl.ToLayer.{LayerPoly1, LayerPoly2}
import com.thoughtworks.deeplearning.dsl.layers.{Compose, Identity, Literal, Throw}

import language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object dsl {

  /** @template */
  type BpNothing = BackPropagationType[Nothing, Any]

  /** @template */
  type BpAny = BackPropagationType[Any, _]

  def `throw`[InputData, InputDelta](throwable: => Throwable)(
      implicit inputType: BackPropagationType[InputData, InputDelta]): Layer.Aux[Batch.Aux[InputData, InputDelta], BpNothing#Batch] = {
    Throw(throwable _)
  }

  implicit def autoToLayer[A, Input <: Batch, OutputData, OutputDelta](a: A)(
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
      managed(toLiteral.forward(Literal[InputData](inputData))).acquireAndGet(_.value)
    }

    def train[InputData, InputDelta](inputData: InputData)(
        implicit ev: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] <:< Layer.Aux[
          Batch.Aux[InputData, InputDelta],
          Batch.Aux[OutputData, OutputDelta]],
        outputDataIsOutputDelta: OutputData <:< OutputDelta
    ): OutputData = {
      val outputBatch = toLiteral.forward(Literal[InputData](inputData))
      try {
        val loss = outputBatch.value
        outputBatch.backward(outputDataIsOutputDelta(loss))
        loss
      } finally {
        outputBatch.close()
      }

    }

  }

  implicit def toAnyOps[A, Input <: Batch, OutputData, OutputDelta](a: A)(
      implicit toLayer: ToLayer.Aux[A, Input, OutputData, OutputDelta]): AnyOps[Input, OutputData, OutputDelta] = {
    new AnyOps(toLayer(a))
  }

  implicit final class ToBatch[Data](a: Data) {
    def toBatch[Delta]: Batch.Aux[Data, Delta] = Literal[Data](a)
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
