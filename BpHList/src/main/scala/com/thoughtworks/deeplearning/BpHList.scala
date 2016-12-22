package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.{Batch, CloseableOnce}
import com.thoughtworks.deeplearning.Lift.BackPropagationType.{DataOf, DeltaOf}
import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.BpHList.Layers._
import shapeless._

import language.implicitConversions
import language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object BpHList {

  object Layers {

    final case class HCons[Input0 <: Batch,
                           HeadData,
                           HeadDelta,
                           TailData <: shapeless.HList,
                           TailDelta <: shapeless.Coproduct](
        head: Layer.Aux[Input0, Batch.Aux[HeadData, HeadDelta]],
        tail: Layer.Aux[Input0, Batch.Aux[TailData, TailDelta]]
    ) extends Layer {
      override type Input = Input0

      final class Output private[HCons] (headBatch: Batch.Aux[HeadData, HeadDelta],
                                         tailBatch: Batch.Aux[TailData, TailDelta])
          extends Batch
          with CloseableOnce {
        override def backward(delta: Delta): Unit = {
          delta match {
            case shapeless.Inl(headDelta) =>
              headBatch.backward(headDelta)
            case shapeless.Inr(tailDelta) =>
              tailBatch.backward(tailDelta)
          }
        }

        override def value: Data = {
          headBatch.value :: tailBatch.value
        }

        override def close(): Unit = {
          super.close()
          headBatch.close()
          tailBatch.close()
        }

        override type Data = HeadData :: TailData
        override type Delta = HeadDelta :+: TailDelta

        override def addReference() = new Output(headBatch.addReference(), tailBatch.addReference())
      }

      override def forward(input: Input) = new Output(head.forward(input), tail.forward(input))

    }

    final case class Head[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList,
    TailDelta <: shapeless.Coproduct](
        operand: Layer.Aux[Input0, Batch.Aux[::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
    ) extends Layer {
      override type Input = Input0

      final class Output private[Head] (
          upstream: Batch.Aux[::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
          extends Batch
          with com.thoughtworks.deeplearning.Layer.CloseableOnce {
        override def backward(delta: Delta): Unit = {
          upstream.backward(shapeless.Inl(delta))
        }

        override def value: Data = {
          upstream.value.head
        }

        override type Data = HeadData
        override type Delta = HeadDelta

        override def close(): Unit = {
          super.close()
          upstream.close()
        }

        override def addReference() = new Output(upstream.addReference())
      }
      override def forward(input: Input) = new Output(operand.forward(input))

    }

    case object HNil extends Layer with Batch {
      override type Input = Batch

      override type Data = shapeless.HNil

      override type Delta = shapeless.CNil

      override type Output = Batch.Aux[Data, Delta]

      override def addReference() = this

      override def forward(input: Input) = this

      override def backward(delta: Delta): Unit = {}

      override def value = shapeless.HNil

      override def close(): Unit = {}
    }

    final case class Tail[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList,
    TailDelta <: shapeless.Coproduct](
        operand: Layer.Aux[Input0, Batch.Aux[HeadData :: TailData, shapeless.:+:[HeadDelta, TailDelta]]]
    ) extends Layer {
      override type Input = Input0

      final class Output private[Tail] (upstream: Batch.Aux[HeadData :: TailData, shapeless.:+:[HeadDelta, TailDelta]])
          extends Batch
          with com.thoughtworks.deeplearning.Layer.CloseableOnce {
        override def backward(delta: Delta): Unit = {
          upstream.backward(shapeless.Inr(delta))
        }

        override def value: Data = {
          upstream.value.tail
        }

        override def close(): Unit = {
          super.close()
          upstream.close()
        }

        override type Data = TailData
        override type Delta = TailDelta

        override def addReference() = new Output(upstream.addReference())
      }
      override def forward(input: Input) = new Output(operand.forward(input))

    }

  }

  /** @template */
  type BpHList = BackPropagationType[_ <: HList, _ <: Coproduct]

  /** @template */
  type BpHNil = BackPropagationType[HNil, CNil]

  /** @template */
  type BpHCons[Head <: BackPropagationType[_, _], Tail <: BpHList] =
    BackPropagationType[DataOf[Head] :: DataOf[Tail], DeltaOf[Head] :+: DeltaOf[Tail]]

  /** @template */
  type :**:[Head <: BackPropagationType[_, _], Tail <: BpHList] =
    BackPropagationType[::[DataOf[Head], DataOf[Tail]], :+:[DeltaOf[Head], DeltaOf[Tail]]]

  val BpHNil: Layers.HNil.type = Layers.HNil

  implicit def hnilToLayer[InputData, InputDelta](implicit inputType: BackPropagationType[InputData, InputDelta])
    : ToLayer.Aux[Layers.HNil.type, Batch.Aux[InputData, InputDelta], HNil, CNil] =
    new ToLayer[Layers.HNil.type, Batch.Aux[InputData, InputDelta]] {
      override type OutputData = HNil
      override type OutputDelta = CNil

      override def apply(hnil: Layers.HNil.type) = hnil
    }

  final class HListLayerOps[Input <: Batch, TailData <: HList, TailDelta <: Coproduct](
      tail: Layer.Aux[Input, Batch.Aux[TailData, TailDelta]]) {

    def ::[Head, HeadData, HeadDelta](head: Head)(implicit headToLayer: ToLayer.Aux[Head, Input, HeadData, HeadDelta])
      : Layer.Aux[Input, Batch.Aux[::[HeadData, TailData], :+:[HeadDelta, TailDelta]]] = {
      HCons[Input, HeadData, HeadDelta, TailData, TailDelta](headToLayer(head), tail)
    }

    def :**:[Head, HeadData, HeadDelta](head: Head)(
        implicit headToLayer: ToLayer.Aux[Head, Input, HeadData, HeadDelta])
      : Layer.Aux[Input, Batch.Aux[::[HeadData, TailData], :+:[HeadDelta, TailDelta]]] = {
      HCons[Input, HeadData, HeadDelta, TailData, TailDelta](headToLayer(head), tail)
    }

  }

  implicit def toHListLayerOps[From, Input <: Batch, TailData <: HList, TailDelta <: Coproduct](from: From)(
      implicit toLayer: ToLayer.Aux[From, Input, TailData, TailDelta]
  ): HListLayerOps[Input, TailData, TailDelta] = {
    new HListLayerOps[Input, TailData, TailDelta](toLayer(from))
  }

  final class HConsLayerOps[Input <: Batch, HeadData, HeadDelta, TailData <: HList, TailDelta <: Coproduct](
      hcons: Layer.Aux[Input, Batch.Aux[::[HeadData, TailData], :+:[HeadDelta, TailDelta]]]) {
    def head: Layer.Aux[Input, Batch.Aux[HeadData, HeadDelta]] =
      Head[Input, HeadData, HeadDelta, TailData, TailDelta](hcons)

    def tail: Layer.Aux[Input, Batch.Aux[TailData, TailDelta]] =
      Tail[Input, HeadData, HeadDelta, TailData, TailDelta](hcons)
  }

  implicit def toHConsLayerOps[From,
                          Input <: Batch,
                          OutputData,
                          OutputDelta,
                          HeadData,
                          HeadDelta,
                          TailData <: HList,
                          TailDelta <: Coproduct](from: From)(
      implicit toLayer: ToLayer.Aux[From, Input, OutputData, OutputDelta],
      toHListLayer: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] <:< Layer.Aux[
        Input,
        Batch.Aux[::[HeadData, TailData], :+:[HeadDelta, TailDelta]]]
  ): HConsLayerOps[Input, HeadData, HeadDelta, TailData, TailDelta] = {
    new HConsLayerOps[Input, HeadData, HeadDelta, TailData, TailDelta](toHListLayer(toLayer(from)))
  }

}
