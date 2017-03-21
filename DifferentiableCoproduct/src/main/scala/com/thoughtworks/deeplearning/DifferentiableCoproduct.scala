package com.thoughtworks.deeplearning

import cats.Eval
import com.thoughtworks.deeplearning.DifferentiableBoolean._
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.DifferentiableBoolean.Layers.If
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.Symbolic.Placeholder.{DataOf, DeltaOf}
import com.thoughtworks.deeplearning.DifferentiableCoproduct.Layers._
import com.thoughtworks.deeplearning.Symbolic.Layers.Literal
import shapeless.{:+:, CNil, Coproduct, Lazy, Lub}

import language.existentials
import language.implicitConversions

/**
  * A namespace of common operators for [[shapeless.Coproduct Coproduct]] layers.
  *
  * After importing `DifferentiableCoproduct._`, the following methods will be available on Coproduct layers.
  *  - [[DifferentiableCoproduct.CConsLayerOps.head head]]
  *  - [[DifferentiableCoproduct.CConsLayerOps.tail tail]]
  *  - [[DifferentiableCoproduct.CConsLayerOps.isInl isInl]]
  *  - [[DifferentiableCoproduct.CConsLayerOps.choice choice]]
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableCoproduct {

  private[deeplearning] type CoproductPlaceholder = Placeholder[_ <: shapeless.Coproduct, _ <: shapeless.Coproduct]

  private[deeplearning] type CNilPlaceholder = Placeholder[shapeless.CNil, shapeless.CNil]

  private[deeplearning] type :++:[Head <: Placeholder[_, _], Tail <: CoproductPlaceholder] =
    Placeholder[shapeless.:+:[DataOf[Head], DataOf[Tail]], shapeless.:+:[DeltaOf[Head], DeltaOf[Tail]]]

  object Layers {

    final case class Head[Input0 <: Tape, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
    TailDelta <: shapeless.Coproduct](
        operand: Layer.Aux[Input0, Tape.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
    ) extends Layer {

      final class Output private[Head] (
          upstream: Tape.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
          extends Tape
          with com.thoughtworks.deeplearning.Layer.CloseableOnce {
        override type Data = HeadData
        override type Delta = HeadDelta

        override val isTrainable = upstream.isTrainable

        override val value =
          upstream.value.asInstanceOf[shapeless.Inl[HeadData, TailData]].head

        override protected def forceBackward(delta: Delta): Unit = {
          upstream.backward(shapeless.Inl(delta))
        }

        override def close(): Unit = {
          super.close()
          upstream.close()
        }

        override def duplicate() = {
          new Output(upstream.duplicate())
        }

      }

      type Input = Input0

      override def forward(input: Input) = new Output(operand.forward(input))

    }

    final case class Inl[Input0 <: Tape, HeadData, HeadDelta](
        operand: Layer.Aux[Input0, Tape.Aux[HeadData, HeadDelta]])
        extends Layer {

      type Input = Input0

      final class Output private[Inl] (upstream: Tape.Aux[HeadData, HeadDelta])
          extends Tape
          with com.thoughtworks.deeplearning.Layer.CloseableOnce {

        override val isTrainable = upstream.isTrainable

        def value = shapeless.Inl(upstream.value: HeadData)

        type Data = shapeless.Inl[HeadData, Nothing]
        type Delta = shapeless.:+:[HeadDelta, shapeless.Coproduct]

        override protected def forceBackward(delta: shapeless.:+:[HeadDelta, shapeless.Coproduct]): Unit = {
          delta match {
            case shapeless.Inl(headDelta) => upstream.backward(headDelta)
            case shapeless.Inr(_) =>
          }
        }

        override def close(): Unit = {
          super.close()
          upstream.close()
        }

        override def duplicate() = new Output(upstream.duplicate())

      }

      override def forward(input: Input) = new Output(operand.forward(input))

    }

    final case class Inr[Input0 <: Tape, TailData <: shapeless.Coproduct, TailDelta <: shapeless.Coproduct](
        operand: Layer.Aux[Input0, Tape.Aux[TailData, TailDelta]])
        extends Layer {

      type Input = Input0

      final class Output private[Inr] (upstream: Tape.Aux[TailData, TailDelta])
          extends Tape
          with com.thoughtworks.deeplearning.Layer.CloseableOnce {

        override val isTrainable = upstream.isTrainable

        override def value = shapeless.Inr(upstream.value: TailData)

        override type Data = shapeless.Inr[Nothing, TailData]
        override type Delta = shapeless.:+:[Any, TailDelta]

        override protected def forceBackward(delta: shapeless.:+:[Any, TailDelta]): Unit = {
          delta match {
            case shapeless.Inr(tailDelta) => upstream.backward(tailDelta)
            case shapeless.Inl(_) =>
          }
        }

        override def close(): Unit = {
          super.close()
          upstream.close()
        }

        override def duplicate() = new Output(upstream.duplicate())

      }

      override def forward(input: Input) = new Output(operand.forward(input))

    }

    final case class Tail[Input0 <: Tape, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
    TailDelta <: shapeless.Coproduct](
        operand: Layer.Aux[Input0, Tape.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
    ) extends Layer {

      final class Output private[Tail] (
          upstream: Tape.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
          extends Tape
          with com.thoughtworks.deeplearning.Layer.CloseableOnce {

        override type Data = TailData
        override type Delta = TailDelta

        override val isTrainable = upstream.isTrainable

        override val value =
          upstream.value.asInstanceOf[shapeless.Inr[TailData, TailData]].tail

        override protected def forceBackward(delta: Delta): Unit = {
          upstream.backward(shapeless.Inr(delta))
        }

        override def close(): Unit = {
          super.close()
          upstream.close()
        }

        override def duplicate() = new Output(upstream.duplicate())
      }

      type Input = Input0

      override def forward(input: Input) = new Output(operand.forward(input))

    }

    final case class IsInl[Input0 <: Tape, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
    TailDelta <: shapeless.Coproduct](
        operand: Layer.Aux[Input0, Tape.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
    ) extends Layer {

      final class Output private[IsInl] (
          upstream: Tape.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
          extends BooleanMonoidTape
          with Tape
          with CloseableOnce {

        override val isTrainable = upstream.isTrainable

        override val value = upstream.value match {
          case shapeless.Inl(_) => true
          case shapeless.Inr(_) => false
        }

        override protected def forceBackward(delta: Boolean): Unit = {}

        override def close(): Unit = {
          super.close()
          upstream.close()
        }

        override def duplicate() = new Output(upstream.duplicate())

      }

      type Input = Input0

      override def forward(input: Input) = new Output(operand.forward(input))
    }

  }

  /**
    * A helper that contains common boilerplate code for all [[shapeless.Coproduct Coproduct]] layers.
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableCoproduct._
    * }}}
    */
  final class CConsLayerOps[
      Input <: Tape,
      HeadData,
      HeadDelta,
      TailData <: shapeless.Coproduct,
      TailDelta <: shapeless.Coproduct
  ](
      ccons: Layer.Aux[
        Input,
        Tape.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]
      ]
  ) {

    def head: Layer.Aux[Input, Tape.Aux[HeadData, HeadDelta]] =
      Head[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)

    def tail: Layer.Aux[Input, Tape.Aux[TailData, TailDelta]] =
      Tail[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)

    def choice[HeadCase,
               TailCase,
               HeadOutputData,
               HeadOutputDelta,
               TailOutputData,
               TailOutputDelta,
               NN,
               OutputData,
               OutputDelta](caseHead: Layer.Aux[Input, Tape.Aux[HeadData, HeadDelta]] => HeadCase)(
        caseTail: Layer.Aux[Input, Tape.Aux[TailData, TailDelta]] => TailCase)(
        implicit headToLayer: ToLayer.Aux[HeadCase, Input, HeadOutputData, HeadOutputDelta],
        tailToLayer: ToLayer.Aux[TailCase, Input, TailOutputData, TailOutputDelta],
        lub: Lub[Layer.Aux[Input, Tape.Aux[HeadOutputData, HeadOutputDelta]],
                 Layer.Aux[Input, Tape.Aux[TailOutputData, TailOutputDelta]],
                 NN],
        commonToLayer: ToLayer.Aux[NN, Input, OutputData, OutputDelta]
    ): Layer.Aux[Input, Tape.Aux[OutputData, OutputDelta]] = {
      If[Input, OutputData, OutputDelta](isInl,
                                         commonToLayer(lub.left(headToLayer(caseHead(head)))),
                                         commonToLayer(lub.right(tailToLayer(caseTail(tail)))))
    }

    def isInl: Layer.Aux[Input, BooleanPlaceholder.Tape] =
      IsInl[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)

  }

  /**
    * Implicitly converts any layer to [[CConsLayerOps]], which enables common methods for CConsLayerOps layers.
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableCoproduct._
    * }}}
    */
  implicit def toCConsLayerOps[From,
                               Input <: Tape,
                               OutputData,
                               OutputDelta,
                               HeadData,
                               HeadDelta,
                               TailData <: shapeless.Coproduct,
                               TailDelta <: shapeless.Coproduct](from: From)(
      implicit toLayer: ToLayer.Aux[From, Input, OutputData, OutputDelta],
      toCoproductLayer: Layer.Aux[Input, Tape.Aux[OutputData, OutputDelta]] <:< Layer.Aux[
        Input,
        Tape.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
  ): CConsLayerOps[Input, HeadData, HeadDelta, TailData, TailDelta] = {
    new CConsLayerOps[Input, HeadData, HeadDelta, TailData, TailDelta](toCoproductLayer(toLayer(from)))
  }

  implicit def cnilToLiteral: ToLiteral.Aux[CNil, CNil, CNil] = ToLiteral.fromData

  implicit def cconsToLiteral[Head,
                              HeadData,
                              HeadDelta,
                              Tail <: Coproduct,
                              TailData <: Coproduct,
                              TailDelta <: Coproduct](
      implicit headToLiteral: Lazy[ToLiteral.Aux[Head, HeadData, HeadDelta]],
      tailToLiteral: Lazy[ToLiteral.Aux[Tail, TailData, TailDelta]])
    : ToLiteral.Aux[Head :+: Tail, HeadData :+: TailData, HeadDelta :+: TailDelta] = new ToLiteral[Head :+: Tail] {
    override type Data = HeadData :+: TailData
    override type Delta = HeadDelta :+: TailDelta
    override def apply(data: :+:[Head, Tail]): Literal[HeadData :+: TailData] = {
      data match {
        case shapeless.Inl(head) =>
          val Literal(headData) = headToLiteral.value(head)
          Literal(shapeless.Inl(headData))
        case shapeless.Inr(tail) =>
          val Literal(tailData) = tailToLiteral.value(tail)
          Literal(shapeless.Inr(tailData))
      }
    }
  }

}
