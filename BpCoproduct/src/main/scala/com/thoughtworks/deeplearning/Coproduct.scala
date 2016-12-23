package com.thoughtworks.deeplearning

import cats.Eval
import com.thoughtworks.deeplearning.BpBoolean._
import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.BpBoolean.Layers.If
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.Lift.Placeholder.{DataOf, DeltaOf}
import com.thoughtworks.deeplearning.BpCoproduct.Layers._
import com.thoughtworks.deeplearning.Lift.Layers.Literal
import shapeless.{:+:, CNil, Coproduct, Lazy, Lub}

import language.existentials
import language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object BpCoproduct {

  /** @template */
  type BpCoproduct = Placeholder[_ <: shapeless.Coproduct, _ <: shapeless.Coproduct]

  /** @template */
  type BpCNil = Placeholder[shapeless.CNil, shapeless.CNil]

  /** @template */
  @deprecated(message="Use `To[Double :+: CNil]` instead",since = "1.0.0")
  type BpCCons[Head <: Placeholder[_, _], Tail <: BpCoproduct] =
    Placeholder[shapeless.:+:[DataOf[Head], DataOf[Tail]], shapeless.:+:[DeltaOf[Head], DeltaOf[Tail]]]

  @deprecated(message="Use `To[Double :+: CNil]` instead",since = "1.0.0")
  type :++:[Head <: Placeholder[_, _], Tail <: BpCoproduct] =
    Placeholder[shapeless.:+:[DataOf[Head], DataOf[Tail]], shapeless.:+:[DeltaOf[Head], DeltaOf[Tail]]]

  object Layers {

    final case class Head[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
    TailDelta <: shapeless.Coproduct](
        operand: Layer.Aux[Input0, Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
    ) extends Layer {

      final class Output private[Head] (
          upstream: Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
          extends Batch
          with com.thoughtworks.deeplearning.Layer.CloseableOnce {
        override type Data = HeadData
        override type Delta = HeadDelta

        val value =
          upstream.value.asInstanceOf[shapeless.Inl[HeadData, TailData]].head

        override def backward(delta: Delta): Unit = {
          upstream.backward(shapeless.Inl(delta))
        }

        override def close(): Unit = {
          super.close()
          upstream.close()
        }

        override def addReference() = {
          new Output(upstream.addReference())
        }

      }

      type Input = Input0

      override def forward(input: Input) = new Output(operand.forward(input))

    }

    final case class Inl[Input0 <: Batch, HeadData, HeadDelta](
        operand: Layer.Aux[Input0, Batch.Aux[HeadData, HeadDelta]])
        extends Layer {

      type Input = Input0

      final class Output private[Inl] (upstream: Batch.Aux[HeadData, HeadDelta])
          extends Batch
          with com.thoughtworks.deeplearning.Layer.CloseableOnce {
        def value = shapeless.Inl(upstream.value: HeadData)

        type Data = shapeless.Inl[HeadData, Nothing]
        type Delta = shapeless.:+:[HeadDelta, shapeless.Coproduct]

        override def backward(delta: shapeless.:+:[HeadDelta, shapeless.Coproduct]): Unit = {
          delta match {
            case shapeless.Inl(headDelta) => upstream.backward(headDelta)
            case shapeless.Inr(_) =>
          }
        }

        override def close(): Unit = {
          super.close()
          upstream.close()
        }

        override def addReference() = new Output(upstream.addReference())

      }

      override def forward(input: Input) = new Output(operand.forward(input))

    }

    final case class Inr[Input0 <: Batch, TailData <: shapeless.Coproduct, TailDelta <: shapeless.Coproduct](
        operand: Layer.Aux[Input0, Batch.Aux[TailData, TailDelta]])
        extends Layer {

      type Input = Input0

      final class Output private[Inr] (upstream: Batch.Aux[TailData, TailDelta])
          extends Batch
          with com.thoughtworks.deeplearning.Layer.CloseableOnce {
        def value = shapeless.Inr(upstream.value: TailData)

        type Data = shapeless.Inr[Nothing, TailData]
        type Delta = shapeless.:+:[Any, TailDelta]

        override def backward(delta: shapeless.:+:[Any, TailDelta]): Unit = {
          delta match {
            case shapeless.Inr(tailDelta) => upstream.backward(tailDelta)
            case shapeless.Inl(_) =>
          }
        }

        override def close(): Unit = {
          super.close()
          upstream.close()
        }

        override def addReference() = new Output(upstream.addReference())

      }

      override def forward(input: Input) = new Output(operand.forward(input))

    }

    final case class Tail[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
    TailDelta <: shapeless.Coproduct](
        operand: Layer.Aux[Input0, Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
    ) extends Layer {

      final class Output private[Tail] (
          upstream: Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
          extends Batch
          with com.thoughtworks.deeplearning.Layer.CloseableOnce {
        override type Data = TailData
        override type Delta = TailDelta

        val value =
          upstream.value.asInstanceOf[shapeless.Inr[TailData, TailData]].tail

        override def backward(delta: Delta): Unit = {
          upstream.backward(shapeless.Inr(delta))
        }

        override def close(): Unit = {
          super.close()
          upstream.close()
        }

        override def addReference() = new Output(upstream.addReference())
      }

      type Input = Input0

      override def forward(input: Input) = new Output(operand.forward(input))

    }

    final case class IsInl[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
    TailDelta <: shapeless.Coproduct](
        operand: Layer.Aux[Input0, Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
    ) extends Layer {

      final class Output private[IsInl] (
          upstream: Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
          extends BooleanMonoidBatch
          with Batch
          with CloseableOnce {

        override val value = upstream.value match {
          case shapeless.Inl(_) => true
          case shapeless.Inr(_) => false
        }

        override def backward(delta: Boolean): Unit = {}

        override def close(): Unit = {
          super.close()
          upstream.close()
        }

        override def addReference() = new Output(upstream.addReference())

      }

      type Input = Input0

      override def forward(input: Input) = new Output(operand.forward(input))
    }

  }

  final class CConsLayerOps[
      Input <: Batch,
      HeadData,
      HeadDelta,
      TailData <: shapeless.Coproduct,
      TailDelta <: shapeless.Coproduct
  ](
      ccons: Layer.Aux[
        Input,
        Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]
      ]
  ) {

    def head: Layer.Aux[Input, Batch.Aux[HeadData, HeadDelta]] =
      Head[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)

    def tail: Layer.Aux[Input, Batch.Aux[TailData, TailDelta]] =
      Tail[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)

    def choice[HeadCase,
               TailCase,
               HeadOutputData,
               HeadOutputDelta,
               TailOutputData,
               TailOutputDelta,
               NN,
               OutputData,
               OutputDelta](caseHead: Layer.Aux[Input, Batch.Aux[HeadData, HeadDelta]] => HeadCase)(
        caseTail: Layer.Aux[Input, Batch.Aux[TailData, TailDelta]] => TailCase)(
        implicit headToLayer: ToLayer.Aux[HeadCase, Input, HeadOutputData, HeadOutputDelta],
        tailToLayer: ToLayer.Aux[TailCase, Input, TailOutputData, TailOutputDelta],
        lub: Lub[Layer.Aux[Input, Batch.Aux[HeadOutputData, HeadOutputDelta]],
                 Layer.Aux[Input, Batch.Aux[TailOutputData, TailOutputDelta]],
                 NN],
        commonToLayer: ToLayer.Aux[NN, Input, OutputData, OutputDelta]
    ): Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = {
      If[Input, OutputData, OutputDelta](isInl,
                                         commonToLayer(lub.left(headToLayer(caseHead(head)))),
                                         commonToLayer(lub.right(tailToLayer(caseTail(tail)))))
    }

    def isInl: Layer.Aux[Input, BooleanPlaceholder.Batch] = IsInl[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)

  }

  implicit def toCConsLayerOps[From,
                               Input <: Batch,
                               OutputData,
                               OutputDelta,
                               HeadData,
                               HeadDelta,
                               TailData <: shapeless.Coproduct,
                               TailDelta <: shapeless.Coproduct](from: From)(
      implicit toLayer: ToLayer.Aux[From, Input, OutputData, OutputDelta],
      toCoproductLayer: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] <:< Layer.Aux[
        Input,
        Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
  ): CConsLayerOps[Input, HeadData, HeadDelta, TailData, TailDelta] = {
    new CConsLayerOps[Input, HeadData, HeadDelta, TailData, TailDelta](toCoproductLayer(toLayer(from)))
  }

  implicit def liftCNil: Lift.Aux[CNil, CNil, CNil] = Lift.fromData[CNil, CNil]

  implicit def liftCCons[Head, HeadData, HeadDelta, Tail <: Coproduct, TailData <: Coproduct, TailDelta <: Coproduct](
      implicit liftHead: Lazy[Lift.Aux[Head, HeadData, HeadDelta]],
      liftTail: Lazy[Lift.Aux[Tail, TailData, TailDelta]])
    : Lift.Aux[Head :+: Tail, HeadData :+: TailData, HeadDelta :+: TailDelta] = new Lift[Head :+: Tail] {
    override type Data = HeadData :+: TailData
    override type Delta = HeadDelta :+: TailDelta
    override def apply(data: :+:[Head, Tail]): Literal[HeadData :+: TailData] = {
      data match {
        case shapeless.Inl(head) =>
          val Literal(headData) = liftHead.value(head)
          Literal(shapeless.Inl(headData))
        case shapeless.Inr(tail) =>
          val Literal(tailData) = liftTail.value(tail)
          Literal(shapeless.Inr(tailData))
      }
    }
  }

}
