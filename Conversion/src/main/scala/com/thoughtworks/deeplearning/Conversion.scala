package com.thoughtworks.deeplearning

import language.implicitConversions
import com.thoughtworks.deeplearning.Layer.Batch
import shapeless._

import language.existentials

// TODO: Rename to Lift
object Conversion {

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

  import Layers._

  object BackPropagationType {

    implicit def apply[Data, Delta]: BackPropagationType[Data, Delta] = new BackPropagationType

    type DataOf[T <: BackPropagationType[_, _]] = t.Data forSome { val t: T }
    type DeltaOf[T <: BackPropagationType[_, _]] = t.Delta forSome { val t: T }

    implicit def inputTypeToLayer[InputData, InputDelta]: ToLayer.Aux[BackPropagationType[InputData, InputDelta],
                                                                      Batch.Aux[InputData, InputDelta],
                                                                      InputData,
                                                                      InputDelta] =
      new ToLayer[BackPropagationType[InputData, InputDelta], Batch.Aux[InputData, InputDelta]] {
        override type OutputData = InputData
        override type OutputDelta = InputDelta

        override def apply(input: BackPropagationType[InputData, InputDelta]) =
          Identity[InputData, InputDelta]()
      }

  }

  trait ToLiteral[From] extends DepFn1[From] {

    type Data
    type Delta

    type Batch = Batch.Aux[Data, Delta]

    type Placeholder = BackPropagationType[Data, Delta]

    type Out = Batch with Layer.Aux[Layer.Batch, Batch]

  }

  object ToLiteral {

    type Aux[From, Data0, Delta0] = ToLiteral[From] {
      type Data = Data0
      type Delta = Delta0
    }

  }

  implicit final class ToLayerOps[From, Input <: Batch, OutputData, OutputDelta](from: From)(
      implicit toLayer: ToLayer.Aux[From, Input, OutputData, OutputDelta]
  ) {

    def toLiteral: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = toLayer(from)

  }

  implicit final class ToLiteralOps[From, Data, Delta](from: From)(
      implicit toLiteral: ToLiteral.Aux[From, Data, Delta]
  ) {

    @inline
    def toBatch: Batch.Aux[Data, Delta] = toLiteral(from)

  }

  final class AnyLayerOps[Input <: Batch, OutputData, OutputDelta](
      layer: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) {}

  implicit def autoToLayer[A, Input <: Batch, OutputData, OutputDelta](a: A)(
      implicit toLayer: ToLayer.Aux[A, Input, OutputData, OutputDelta])
    : Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = {
    toLayer(a)
  }

  private[deeplearning] sealed trait ToLayerLowPriorityImplicits { this: ToLayer.type =>

    implicit def toLayerOfType[Input0 <: Batch, OutputType <: BackPropagationType[_, _]]
      : ToLayer.OfType[Layer.Aux[Input0, OutputType#Batch], Input0, OutputType] = {
      ToLayer
        .layerToLayer[Input0, BackPropagationType.DataOf[OutputType], BackPropagationType.DeltaOf[OutputType]]
        .asInstanceOf[ToLayer.OfType[Layer.Aux[Input0, OutputType#Batch], Input0, OutputType]]
    }

  }

  // FIXME: rename to placeholder
  final class BackPropagationType[Data0, Delta0] {
    type Data = Data0
    type Delta = Delta0

    private type ConcreteBatch = Batch.Aux[Data, Delta]

    // Workaround for https://issues.scala-lang.org/browse/SI-10008
    type Batch >: ConcreteBatch <: ConcreteBatch

    @deprecated(since = "1.0.0",
                message = "Use Layer.Aux[the.`ToLiteral[InputData]`.Out, the.`ToLiteral[OutputData]`.Out] instead")
    type To[OutputSymbol <: BackPropagationType[_, _]] = Layer.Aux[Batch, OutputSymbol#Batch]
    //  type Layer.Aux[OutputType <: BackPropagationType] =
    //    Layer.Aux[ConcreteBatch, outputType.ConcreteBatch forSome { val outputType: OutputType }]
  }

  object ToLayer extends ToLayerLowPriorityImplicits {

    type Aux[From, Input <: Batch, OutputData0, OutputDelta0] = ToLayer[From, Input] {
      type OutputData = OutputData0
      type OutputDelta = OutputDelta0
    }

    type OfType[From, Input <: Batch, OutputType <: BackPropagationType[_, _]] =
      ToLayer.Aux[From, Input, differentiableType.Data, differentiableType.Delta] forSome {
        val differentiableType: OutputType
      }

    implicit def layerToLayer[Input <: Batch, OutputData0, OutputDelta0]
      : ToLayer.Aux[Layer.Aux[Input, Batch.Aux[OutputData0, OutputDelta0]], Input, OutputData0, OutputDelta0] =
      new ToLayer[Layer.Aux[Input, Batch.Aux[OutputData0, OutputDelta0]], Input] {
        override type OutputData = OutputData0
        override type OutputDelta = OutputDelta0

        override def apply(layer: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) = layer
      }

    implicit def literalToLayer[From, InputData, InputDelta, OutputData0, OutputDelta0](
        implicit inputType: BackPropagationType[InputData, InputDelta],
        toLiteral: ToLiteral.Aux[From, OutputData0, OutputDelta0])
      : ToLayer.Aux[From, Batch.Aux[InputData, InputDelta], OutputData0, OutputDelta0] = {
      new ToLayer[From, Batch.Aux[InputData, InputDelta]] {
        override type OutputData = OutputData0
        override type OutputDelta = OutputDelta0

        override def apply(from: From) = toLiteral(from)
      }
    }

  }

  /**
    * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
    */
  trait ToLayer[From, Input <: Batch] extends DepFn1[From] {
    type OutputData
    type OutputDelta
    type Output = Batch.Aux[OutputData, OutputDelta]
    type Out = Layer.Aux[Input, Output]
  }

  trait Parameter[NativeValue] {

    type Data
    type Delta

    type Out = BackPropagationType[Data, Delta]

  }

  object Parameter {

    /** @template */
    type Aux[NativeValue, Data0, Delta0] = Parameter[NativeValue] {
      type Data = Data0
      type Delta = Delta0
    }

    def apply[NativeValue, Data, Delta](
        implicit typeClass: Parameter.Aux[NativeValue, Data, Delta]): Parameter.Aux[NativeValue, Data, Delta] =
      typeClass

    implicit def fromLift[NativeValue, Data0, Delta0](
        implicit lift: Lazy[ToLiteral.Aux[NativeValue, Data0, Delta0]]): Parameter.Aux[NativeValue, Data0, Delta0] =
      new Parameter[NativeValue] {
        type Data = Data0
        type Delta = Delta0
      }

  }
  object <=> {

    /** @template */
    type Aux[NativeInput, NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0] =
      <=>[NativeInput, NativeOutput] {
        type InputData = InputData0
        type InputDelta = InputDelta0
        type OutputData = OutputData0
        type OutputDelta = OutputDelta0
      }

    def apply[NativeInput, NativeOutput, InputData, InputDelta, OutputData, OutputDelta](
        implicit typeClass: <=>.Aux[NativeInput, NativeOutput, InputData, InputDelta, OutputData, OutputDelta])
      : <=>.Aux[NativeInput, NativeOutput, InputData, InputDelta, OutputData, OutputDelta] = typeClass

    implicit def fromBatchOf[NativeInput, NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0](
        implicit inputBatchOf: Lazy[ToLiteral.Aux[NativeInput, InputData0, InputDelta0]],
        outputBatchOf: Lazy[ToLiteral.Aux[NativeOutput, OutputData0, OutputDelta0]])
      : <=>.Aux[NativeInput, NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0] =
      new <=>[NativeInput, NativeOutput] {
        type InputData = InputData0
        type InputDelta = InputDelta0
        type OutputData = OutputData0
        type OutputDelta = OutputDelta0
      }

  }

  trait <=>[NativeInput, NativeOutput] {
    type InputData
    type InputDelta
    type OutputData
    type OutputDelta
    type Input = Batch.Aux[InputData, InputDelta]
    type Output = Batch.Aux[OutputData, OutputDelta]
    type Out = Layer.Aux[Input, Output]
  }

}
