package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.{Tape, CloseableOnce}
import com.thoughtworks.deeplearning.Symbolic.Placeholder.{DataOf, DeltaOf}
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.DifferentiableHList.Layers._
import com.thoughtworks.deeplearning.Symbolic.Layers.Literal
import shapeless._

import language.implicitConversions
import language.existentials

/**
  * A namespace of common operators for [[shapeless.HList HList]] layers.
  *
  * After importing `DifferentiableHList._`, the following methods will be available on HList layers.
  *  - [[DifferentiableHList.HListLayerOps.:: ::]]
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableHList {

  object Layers {

    final case class HCons[Input0 <: Tape,
                           HeadData,
                           HeadDelta,
                           TailData <: shapeless.HList,
                           TailDelta <: shapeless.Coproduct](
        head: Layer.Aux[Input0, Tape.Aux[HeadData, HeadDelta]],
        tail: Layer.Aux[Input0, Tape.Aux[TailData, TailDelta]]
    ) extends Layer {
      override type Input = Input0

      final class Output private[HCons] (headTape: Tape.Aux[HeadData, HeadDelta],
                                         tailTape: Tape.Aux[TailData, TailDelta])
          extends Tape
          with CloseableOnce {

        override val isTrainable = headTape.isTrainable || tailTape.isTrainable

        override protected def forceBackward(delta: Delta): Unit = {
          delta match {
            case shapeless.Inl(headDelta) =>
              headTape.backward(headDelta)
            case shapeless.Inr(tailDelta) =>
              tailTape.backward(tailDelta)
          }
        }

        override def value: Data = {
          headTape.value :: tailTape.value
        }

        override def close(): Unit = {
          super.close()
          headTape.close()
          tailTape.close()
        }

        override type Data = HeadData :: TailData
        override type Delta = HeadDelta :+: TailDelta

        override def duplicate() = new Output(headTape.duplicate(), tailTape.duplicate())
      }

      override def forward(input: Input) = new Output(head.forward(input), tail.forward(input))

    }

    final case class Head[Input0 <: Tape, HeadData, HeadDelta, TailData <: shapeless.HList,
    TailDelta <: shapeless.Coproduct](
        operand: Layer.Aux[Input0, Tape.Aux[::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
    ) extends Layer {
      override type Input = Input0

      final class Output private[Head] (
          upstream: Tape.Aux[::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
          extends Tape
          with com.thoughtworks.deeplearning.Layer.CloseableOnce {

        override val isTrainable = upstream.isTrainable

        override protected def forceBackward(delta: Delta): Unit = {
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

        override def duplicate() = new Output(upstream.duplicate())
      }
      override def forward(input: Input) = new Output(operand.forward(input))

    }

    final case class Tail[Input0 <: Tape, HeadData, HeadDelta, TailData <: shapeless.HList,
    TailDelta <: shapeless.Coproduct](
        operand: Layer.Aux[Input0, Tape.Aux[HeadData :: TailData, shapeless.:+:[HeadDelta, TailDelta]]]
    ) extends Layer {
      override type Input = Input0

      final class Output private[Tail] (upstream: Tape.Aux[HeadData :: TailData, shapeless.:+:[HeadDelta, TailDelta]])
          extends Tape
          with com.thoughtworks.deeplearning.Layer.CloseableOnce {

        override val isTrainable = upstream.isTrainable

        override protected def forceBackward(delta: Delta): Unit = {
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

        override def duplicate() = new Output(upstream.duplicate())
      }
      override def forward(input: Input) = new Output(operand.forward(input))

    }

  }

  private[deeplearning] type HListPlaceholder = Placeholder[_ <: HList, _ <: Coproduct]

  private[deeplearning] type HNilPlaceholder = Placeholder[HNil, CNil]

  private[deeplearning] type :**:[Head <: Placeholder[_, _], Tail <: HListPlaceholder] =
    Placeholder[::[DataOf[Head], DataOf[Tail]], :+:[DeltaOf[Head], DeltaOf[Tail]]]

  final class HListLayerOps[Input <: Tape, TailData <: HList, TailDelta <: Coproduct](
      tail: Layer.Aux[Input, Tape.Aux[TailData, TailDelta]]) {

    def ::[Head, HeadData, HeadDelta](head: Head)(implicit headToLayer: ToLayer.Aux[Head, Input, HeadData, HeadDelta])
      : Layer.Aux[Input, Tape.Aux[::[HeadData, TailData], :+:[HeadDelta, TailDelta]]] = {
      HCons[Input, HeadData, HeadDelta, TailData, TailDelta](headToLayer(head), tail)
    }

  }

  /**
    * Implicitly converts any layer to [[HListLayerOps]], which enables common methods for HList layers.
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableHList._
    * }}}
    */
  implicit def toHListLayerOps[From, Input <: Tape, TailData <: HList, TailDelta <: Coproduct](from: From)(
      implicit toLayer: ToLayer.Aux[From, Input, TailData, TailDelta]
  ): HListLayerOps[Input, TailData, TailDelta] = {
    new HListLayerOps[Input, TailData, TailDelta](toLayer(from))
  }

  final class HConsLayerOps[Input <: Tape, HeadData, HeadDelta, TailData <: HList, TailDelta <: Coproduct](
      hcons: Layer.Aux[Input, Tape.Aux[::[HeadData, TailData], :+:[HeadDelta, TailDelta]]]) {
    def head: Layer.Aux[Input, Tape.Aux[HeadData, HeadDelta]] =
      Head[Input, HeadData, HeadDelta, TailData, TailDelta](hcons)

    def tail: Layer.Aux[Input, Tape.Aux[TailData, TailDelta]] =
      Tail[Input, HeadData, HeadDelta, TailData, TailDelta](hcons)
  }

  implicit def toHConsLayerOps[From,
                               Input <: Tape,
                               OutputData,
                               OutputDelta,
                               HeadData,
                               HeadDelta,
                               TailData <: HList,
                               TailDelta <: Coproduct](from: From)(
      implicit toLayer: ToLayer.Aux[From, Input, OutputData, OutputDelta],
      toHListLayer: Layer.Aux[Input, Tape.Aux[OutputData, OutputDelta]] <:< Layer.Aux[
        Input,
        Tape.Aux[::[HeadData, TailData], :+:[HeadDelta, TailDelta]]]
  ): HConsLayerOps[Input, HeadData, HeadDelta, TailData, TailDelta] = {
    new HConsLayerOps[Input, HeadData, HeadDelta, TailData, TailDelta](toHListLayer(toLayer(from)))
  }

  implicit def hnilToLiteral[From <: HNil]: ToLiteral.Aux[From, HNil, CNil] = ToLiteral.fromData

  implicit def hconsToLiteral[Head, HeadData, HeadDelta, Tail <: HList, TailData <: HList, TailDelta <: Coproduct](
      implicit headToLiteral: Lazy[ToLiteral.Aux[Head, HeadData, HeadDelta]],
      tailToLiteral: Lazy[ToLiteral.Aux[Tail, TailData, TailDelta]])
    : ToLiteral.Aux[Head :: Tail, HeadData :: TailData, HeadDelta :+: TailDelta] = new ToLiteral[Head :: Tail] {
    override type Data = HeadData :: TailData
    override type Delta = HeadDelta :+: TailDelta
    override def apply(data: Head :: Tail): Literal[HeadData :: TailData] = {
      val head :: tail = data
      val Literal(headData) = headToLiteral.value(head)
      val Literal(tailData) = tailToLiteral.value(tail)
      Literal(headData :: tailData)
    }
  }
}
