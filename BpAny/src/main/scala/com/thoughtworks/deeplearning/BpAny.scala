package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.BpAny.Layers.Compose
import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.Conversion.Layers.Literal
import com.thoughtworks.deeplearning.Conversion._
import resource.managed
import language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object BpAny {

  /** @template */
  type BpAny = BackPropagationType[Any, _]

  object Layers {

    final case class Compose[Input0 <: Batch, Temporary <: Batch, Output0 <: Batch](
        leftOperand: Layer.Aux[Temporary, Output0],
        rightOperand: Layer.Aux[Input0, Temporary])
        extends Layer {
      override type Input = Input0
      override type Output = Output0

      override def forward(input: Input): Output = {
        val tmpBatch = rightOperand.forward(input)
        try {
          leftOperand.forward(tmpBatch)
        } finally {
          tmpBatch.close()
        }
      }
    }

  }

  final class BpAnyOps[Input <: Batch, OutputData, OutputDelta](
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

  implicit def toBpAnyOps[A, Input <: Batch, OutputData, OutputDelta](a: A)(
      implicit toLayer: ToLayer.Aux[A, Input, OutputData, OutputDelta]): BpAnyOps[Input, OutputData, OutputDelta] = {
    new BpAnyOps(toLayer(a))
  }

}
