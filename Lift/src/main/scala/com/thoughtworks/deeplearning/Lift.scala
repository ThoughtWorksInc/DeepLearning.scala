package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.Lift.Layers.Identity
import shapeless.DepFn1

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait Lift[NativeValue] extends DepFn1[NativeValue] {

  type Data
  type Delta

  type Batch = Batch.Aux[Data, Delta]

  type Out = Batch with Layer.Aux[Layer.Batch, Batch]

}

object Lift {

  type Aux[NativeValue, Data0, Delta0] = Lift[NativeValue] {
    type Data = Data0
    type Delta = Delta0
  }

  object Layers {

    final case class Identity[Data, Delta]() extends Layer {
      type Input = Batch.Aux[Data, Delta]
      type Output = Batch.Aux[Data, Delta]

      override def forward(input: Input): Output = {
        input.addReference()
      }
    }

    final case class Literal[Data0](value0: Data0) extends Layer with Batch {
      override type Data = Data0
      override type Delta = Any
      override type Input = Batch
      override type Output = Batch.Aux[Data, Delta]

      override def value: Data = value0

      override def forward(input: Input) = this

      override def backward(delta: Delta): Unit = {}

      override def close(): Unit = {}

      override def addReference() = this
    }

  }

  implicit def placeholder[Data, Delta]: Identity[Data, Delta] = new Identity[Data, Delta]

  // TODO: Use InputData and InputDelta, instead of a single type parameter Input
  trait ToLayer[NativeValue, Input <: Batch] extends DepFn1[NativeValue] {
    type OutputData
    type OutputDelta
    type Output = Batch.Aux[OutputData, OutputDelta]
    type Out = Layer.Aux[Input, Output]
  }

  object ToLayer {

    type Aux[From, Input <: Batch, OutputData0, OutputDelta0] = ToLayer[From, Input] {
      type OutputData = OutputData0
      type OutputDelta = OutputDelta0
    }

    implicit def layerToLayer[Input <: Batch, OutputData0, OutputDelta0]
      : ToLayer.Aux[Layer.Aux[Input, Batch.Aux[OutputData0, OutputDelta0]], Input, OutputData0, OutputDelta0] =
      new ToLayer[Layer.Aux[Input, Batch.Aux[OutputData0, OutputDelta0]], Input] {
        override type OutputData = OutputData0
        override type OutputDelta = OutputDelta0

        override def apply(layer: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) = layer
      }

    implicit def liftToLayer[From, InputData, InputDelta, OutputData0, OutputDelta0](
        implicit currentParameter: Identity[InputData, InputDelta],
        lift: Lift.Aux[From, OutputData0, OutputDelta0])
      : ToLayer.Aux[From, Batch.Aux[InputData, InputDelta], OutputData0, OutputDelta0] = {
      new ToLayer[From, Batch.Aux[InputData, InputDelta]] {
        override type OutputData = OutputData0
        override type OutputDelta = OutputDelta0
        override def apply(from: From) = lift(from)
      }
    }
  }

  implicit final class LiftOps[NativeValue, Data, Delta](value: NativeValue)(
      implicit lift: Lift.Aux[NativeValue, Data, Delta]) {
    def toBatch: Batch.Aux[Data, Delta] = lift(value)
  }

  trait LayerOf[NativeInput, NativeOutput] {
    type InputData
    type InputDelta
    type OutputData
    type OutputDelta
    type Input = Batch.Aux[InputData, InputDelta]
    type Output = Batch.Aux[OutputData, OutputDelta]
    type Out = Layer.Aux[Input, Output]
  }

  trait BatchOf[NativeValue] {

    type Data
    type Delta

    type Out = Batch.Aux[Data, Delta]

  }

  object BatchOf {

    /** @template */
    type Aux[NativeValue, Data0, Delta0] = BatchOf[NativeValue] {
      type Data = Data0
      type Delta = Delta0
    }

    def apply[NativeValue, Data, Delta](
        implicit typeClass: BatchOf.Aux[NativeValue, Data, Delta]): BatchOf.Aux[NativeValue, Data, Delta] =
      typeClass

    implicit def fromLift[NativeValue, Data0, Delta0](
        implicit lift: Lift.Aux[NativeValue, Data0, Delta0]): BatchOf.Aux[NativeValue, Data0, Delta0] =
      new BatchOf[NativeValue] {
        type Data = Data0
        type Delta = Delta0
      }

  }

  object LayerOf {

    /** @template */
    type Aux[NativeInput, NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0] =
      LayerOf[NativeInput, NativeOutput] {
        type InputData = InputData0
        type InputDelta = InputDelta0
        type OutputData = OutputData0
        type OutputDelta = OutputDelta0
      }

    def apply[NativeInput, NativeOutput, InputData, InputDelta, OutputData, OutputDelta](
        implicit typeClass: LayerOf.Aux[NativeInput, NativeOutput, InputData, InputDelta, OutputData, OutputDelta])
      : LayerOf.Aux[NativeInput, NativeOutput, InputData, InputDelta, OutputData, OutputDelta] = typeClass

    implicit def fromBatchOf[NativeInput, NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0](
        implicit inputBatchOf: BatchOf.Aux[NativeInput, InputData0, InputDelta0],
        outputBatchOf: BatchOf.Aux[NativeOutput, OutputData0, OutputDelta0])
      : LayerOf.Aux[NativeInput, NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0] =
      new LayerOf[NativeInput, NativeOutput] {
        type InputData = InputData0
        type InputDelta = InputDelta0
        type OutputData = OutputData0
        type OutputDelta = OutputDelta0
      }

  }

}
