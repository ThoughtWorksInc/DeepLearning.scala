package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.{Aux, Batch}
import com.thoughtworks.deeplearning.Symbolic.Layers.Literal
import shapeless._

import scala.annotation.implicitNotFound
import scala.language.{existentials, implicitConversions}

@implicitNotFound("Don't know how to make ${NativeOutput} differentiable")
trait Symbolic[NativeOutput] {
  type T
}

private[deeplearning] trait LowPrioritySymbolic { this: Symbolic.type =>

  implicit def fromLiteral[NativeOutput, Data0, Delta0](
      implicit toLiteral: Lazy[ToLiteral.Aux[NativeOutput, Data0, Delta0]]): From.Aux[NativeOutput, Data0, Delta0] =
    new From[NativeOutput] {
      type Data = Data0
      type Delta = Delta0
    }

}

object Symbolic extends LowPrioritySymbolic {

  trait ToLiteral[From] extends DepFn1[From] {

    type Data
    type Delta

    type Out = Literal[Data]

  }

  object ToLiteral {

    def fromData[From <: Data0, Data0, Delta0] = new ToLiteral[From] {
      override type Data = Data0
      override type Delta = Delta0

      override def apply(data: From) = Literal[Data](data)
    }

    type Aux[From, Data0, Delta0] = ToLiteral[From] {
      type Data = Data0
      type Delta = Delta0
    }

  }

  object Layers {

    final case class Identity[Data0, Delta0]() extends Layer {

      type Data = Data0
      type Delta = Delta0

      type Input = Batch.Aux[Data, Delta]
      type Output = Batch.Aux[Data, Delta]

      override def forward(input: Input): Output = {
        input.addReference()
      }

      private type ConcreteBatch = Batch.Aux[Data, Delta]

      // Workaround for https://issues.scala-lang.org/browse/SI-10008
      type Batch >: ConcreteBatch <: ConcreteBatch

      private[deeplearning] type To[OutputPlaceholder <: Identity[_, _]] = Layer.Aux[Batch, OutputPlaceholder#Batch]

    }

    object Identity {

      implicit def implicitlyApply[Data, Delta]: Identity[Data, Delta] = new Identity

      private[deeplearning] type DataOf[T <: Identity[_, _]] = t.Data forSome { val t: T }
      private[deeplearning] type DeltaOf[T <: Identity[_, _]] = t.Delta forSome { val t: T }

      implicit def inputPlaceholderToLayer[InputData, InputDelta]
        : ToLayer.Aux[Identity[InputData, InputDelta], Batch.Aux[InputData, InputDelta], InputData, InputDelta] =
        new ToLayer[Identity[InputData, InputDelta], Batch.Aux[InputData, InputDelta]] {
          override type OutputData = InputData
          override type OutputDelta = InputDelta

          override def apply(input: Identity[InputData, InputDelta]) =
            Identity[InputData, InputDelta]()
        }

    }

    final case class Literal[Data0](value0: Data0) extends Layer with Batch {
      override type Data = Data0
      override type Delta = Any
      override type Input = Batch
      override type Output = Batch.Aux[Data, Delta]

      override def value: Data = value0

      override def forward(input: Input) = this

      override def isTrainable: Boolean = false

      override protected def forceBackward(delta: Delta): Unit = {}

      override def close(): Unit = {}

      override def addReference() = this
    }

  }

  import Layers._

  private[deeplearning] trait IsLayer {
    type OutputData
    type OutputDelta
    type InputData
    type InputDelta
    type ConcreteLayer = Layer.Aux[Batch.Aux[InputData, InputDelta], Batch.Aux[OutputData, OutputDelta]]
    type T >: ConcreteLayer <: ConcreteLayer
  }

  private[deeplearning] object IsLayer {

    type Aux[InputData0, InputDelta0, OutputData0, OutputDelta0] = IsLayer {
      type OutputData = OutputData0
      type OutputDelta = OutputDelta0
      type InputData = InputData0
      type InputDelta = InputDelta0
    }

  }

  private[deeplearning] trait To[NativeOutput] extends Symbolic[NativeOutput] with IsLayer

  private[deeplearning] object To {
    type Aux[NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0] = To[NativeOutput] {
      type OutputData = OutputData0
      type OutputDelta = OutputDelta0
      type InputData = InputData0
      type InputDelta = InputDelta0
    }

    def apply[NativeOutput](implicit tc: To[NativeOutput]): tc.type = tc
  }

  implicit def to[NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0](
      implicit inputPlaceHolder: Identity[InputData0, InputDelta0],
      liftTo: ToLiteral.Aux[NativeOutput, OutputData0, OutputDelta0]
  ): To.Aux[NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0] =
    new To[NativeOutput] {
      type OutputData = OutputData0
      type OutputDelta = OutputDelta0
      type InputData = InputData0
      type InputDelta = InputDelta0
    }

  trait LayerOf[NativeInput, NativeOutput] extends IsLayer

  object LayerOf {

    /** @template */
    type Aux[NativeInput, NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0] =
      LayerOf[NativeInput, NativeOutput] {
        type InputData = InputData0
        type InputDelta = InputDelta0
        type OutputData = OutputData0
        type OutputDelta = OutputDelta0
      }

    def apply[NativeInput, NativeOutput](implicit typeClass: LayerOf[NativeInput, NativeOutput]): typeClass.type =
      typeClass

    implicit def lift[NativeInput, NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0](
        implicit liftInput: Lazy[ToLiteral.Aux[NativeInput, InputData0, InputDelta0]],
        liftOutput: Lazy[ToLiteral.Aux[NativeOutput, OutputData0, OutputDelta0]])
      : LayerOf.Aux[NativeInput, NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0] =
      new LayerOf[NativeInput, NativeOutput] {
        type InputData = InputData0
        type InputDelta = InputDelta0
        type OutputData = OutputData0
        type OutputDelta = OutputDelta0
      }

  }

  type <=>[NativeInput, NativeOutput] = LayerOf[NativeInput, NativeOutput]

  private[deeplearning] type Placeholder[Data, Delta] = Identity[Data, Delta]

  private[deeplearning] val Placeholder = Identity

  implicit final class ToLayerOps[From, Input <: Batch, OutputData, OutputDelta](from: From)(
      implicit typeClassInstance: ToLayer.Aux[From, Input, OutputData, OutputDelta]
  ) {

    def toLayer: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = typeClassInstance(from)

  }

  implicit final class ToBatchOps[From, Data, Delta](from: From)(
      implicit lift: ToLiteral.Aux[From, Data, Delta]
  ) {

    @inline
    def toBatch: Batch.Aux[Data, Delta] = lift(from)

  }

  implicit def autoToLayer[A, Input <: Batch, OutputData, OutputDelta](a: A)(
      implicit toLayer: ToLayer.Aux[A, Input, OutputData, OutputDelta])
    : Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = {
    toLayer(a)
  }

  private[deeplearning] sealed trait ToLayerLowPriorityImplicits { this: ToLayer.type =>

    implicit def toLayerOfPlaceholder[Input0 <: Batch, OutputPlaceholder <: Identity[_, _]]
      : ToLayer.OfPlaceholder[Layer.Aux[Input0, OutputPlaceholder#Batch], Input0, OutputPlaceholder] = {
      ToLayer
        .layerToLayer[Input0, Placeholder.DataOf[OutputPlaceholder], Placeholder.DeltaOf[OutputPlaceholder]]
        .asInstanceOf[ToLayer.OfPlaceholder[Layer.Aux[Input0, OutputPlaceholder#Batch], Input0, OutputPlaceholder]]
    }

    implicit def isLayerToLayer[NativeInput, NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0]
      : ToLayer.Aux[
        IsLayer.Aux[InputData0, InputDelta0, OutputData0, OutputDelta0]#T,
        Batch.Aux[InputData0, InputDelta0],
        OutputData0,
        OutputDelta0
      ] = {
      layerToLayer
    }

  }

  object ToLayer extends ToLayerLowPriorityImplicits {

    type Aux[From, Input <: Batch, OutputData0, OutputDelta0] = ToLayer[From, Input] {
      type OutputData = OutputData0
      type OutputDelta = OutputDelta0
    }

    type OfPlaceholder[From, Input <: Batch, OutputPlaceholder <: Identity[_, _]] =
      ToLayer.Aux[From, Input, differentiablePlaceholder.Data, differentiablePlaceholder.Delta] forSome {
        val differentiablePlaceholder: OutputPlaceholder
      }

    implicit def layerToLayer[Input <: Batch, OutputData0, OutputDelta0]
      : ToLayer.Aux[Layer.Aux[Input, Batch.Aux[OutputData0, OutputDelta0]], Input, OutputData0, OutputDelta0] =
      new ToLayer[Layer.Aux[Input, Batch.Aux[OutputData0, OutputDelta0]], Input] {
        override type OutputData = OutputData0
        override type OutputDelta = OutputDelta0

        override def apply(layer: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) = layer
      }

    implicit def placeholderToLayer[From, InputData, InputDelta, OutputData0, OutputDelta0](
        implicit inputPlaceholder: Identity[InputData, InputDelta],
        lift: ToLiteral.Aux[From, OutputData0, OutputDelta0])
      : ToLayer.Aux[From, Batch.Aux[InputData, InputDelta], OutputData0, OutputDelta0] = {
      new ToLayer[From, Batch.Aux[InputData, InputDelta]] {
        override type OutputData = OutputData0
        override type OutputDelta = OutputDelta0

        override def apply(from: From) = lift(from)
      }
    }

  }

  /**
    * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
    */
  @implicitNotFound("Cannot convert ${From} to layer")
  trait ToLayer[From, Input <: Batch] extends DepFn1[From] {
    type OutputData
    type OutputDelta
    type Output = Batch.Aux[OutputData, OutputDelta]
    type Out = Layer.Aux[Input, Output]
  }

  private[deeplearning] trait From[NativeOutput] extends Symbolic[NativeOutput] with DepFn0 {

    type Data
    type Delta

    type T = Identity[Data, Delta]

    type Out = T

    override def apply() = new Identity

  }

  private[deeplearning] object From {

    /** @template */
    type Aux[NativeOutput, Data0, Delta0] = From[NativeOutput] {
      type Data = Data0
      type Delta = Delta0
    }

    def apply[NativeOutput](implicit typeClass: From[NativeOutput]): typeClass.type = typeClass

  }

}
