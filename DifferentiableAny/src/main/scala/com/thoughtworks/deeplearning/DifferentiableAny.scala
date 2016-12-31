package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.DifferentiableAny.Layers.Compose
import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.Lift.Layers.Literal
import com.thoughtworks.deeplearning.Lift._
import resource.managed
import language.implicitConversions
import language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableAny {

  private[deeplearning] type AnyPlaceholder = Placeholder[Any, ExistentialNothing]
  private[deeplearning] val AnyPlaceholder: AnyPlaceholder = implicitly

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

  final class AnyLayerOps[Input <: Batch, OutputData, OutputDelta](
      layer: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) {

    def compose[G, NewInput <: Batch, InputData, InputDelta](g: G)(
        implicit toLayer: ToLayer.Aux[G, NewInput, InputData, InputDelta],
        toInput: Layer.Aux[NewInput, Batch.Aux[InputData, InputDelta]] <:< Layer.Aux[NewInput, Input]
    ): Layer.Aux[NewInput, Batch.Aux[OutputData, OutputDelta]] = {
      Compose(layer, toInput(toLayer(g)))
    }

    def predict[InputData, InputDelta](inputData: InputData)(
        implicit ev: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] <:< Layer.Aux[
          Batch.Aux[InputData, InputDelta],
          Batch.Aux[OutputData, OutputDelta]]
    ): OutputData = {
      managed(layer.forward(Literal[InputData](inputData))).acquireAndGet(_.value)
    }

    def train[InputData, InputDelta](inputData: InputData)(
        implicit ev: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] <:< Layer.Aux[
          Batch.Aux[InputData, InputDelta],
          Batch.Aux[OutputData, OutputDelta]],
        outputDataIsOutputDelta: Trainable[OutputData, OutputDelta]
    ): OutputData = {
      val outputBatch = layer.forward(Literal[InputData](inputData))
      try {
        val loss = outputBatch.value
        outputBatch.backward(outputDataIsOutputDelta(loss))
        loss
      } finally {
        outputBatch.close()
      }

    }

  }

  implicit def toAnyLayerOps[A, Input <: Batch, OutputData, OutputDelta](a: A)(
      implicit toLayer: ToLayer.Aux[A, Input, OutputData, OutputDelta])
    : AnyLayerOps[Input, OutputData, OutputDelta] = {
    new AnyLayerOps(toLayer(a))
  }

  type ExistentialNothing = T forSome { type T >: Nothing <: Nothing }

  implicit def liftAny: Lift.Aux[Any, Any, ExistentialNothing] = Lift.fromData

  trait Trainable[-Data, +Delta] {
    def apply(data: Data): Delta
  }

}
