package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.{Tape, CloseableOnce}
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.DifferentiableSeq.Layers.{Get, ToSeq}
import com.thoughtworks.deeplearning.Symbolic.Layers.Literal
import shapeless.Lazy

import language.implicitConversions
import language.higherKinds

/**
  * A namespace of common operators for Seq layers.
  *
  * After importing `DifferentiableSeq._`, the following methods will be available on Seq layers.
  *  - [[DifferentiableSeq.SeqLayerOps.apply apply]]
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableSeq {

  object Layers {

    final case class Get[Input0 <: Tape, ElementData, ElementDelta](
        operand0: Layer.Aux[Input0, Tape.Aux[Seq[ElementData], (Int, ElementDelta)]],
        i: Int
    ) extends Layer {

      final class Output private[Get] (upstream: Tape.Aux[Seq[ElementData], (Int, ElementDelta)]) extends Tape {

        override val isTrainable = upstream.isTrainable

        type Delta = ElementDelta
        type Data = ElementData
        override protected def forceBackward(delta: ElementDelta): Unit = {
          upstream.backward((i, delta))
        }

        override def duplicate() = new Output(upstream.duplicate())

        override def close(): Unit = {
          upstream.close()
        }

        override val value = {
          upstream.value(i)
        }

      }
      override type Input = Input0

      // TODO: Support tail Int
      override def forward(input: Input) = new Output(operand0.forward(input))

    }

    final case class ToSeq[Input0 <: Tape, ElementData, ElementDelta](
        operands: Seq[Layer.Aux[Input0, Tape.Aux[ElementData, ElementDelta]]])
        extends Layer {

      type Input = Input0

      final class Output private[ToSeq] (upstreams: Seq[Tape.Aux[ElementData, ElementDelta]])
          extends Tape
          with CloseableOnce {

        override type Data = Seq[ElementData]
        override type Delta = (Int, ElementDelta)

        override val isTrainable = upstreams.exists(_.isTrainable)

        override protected def forceBackward(pair: (Int, ElementDelta)): Unit = {
          val (i, delta) = pair
          upstreams(i).backward(delta)
        }

        override val value = {
          upstreams.map(_.value)
        }

        override def close(): Unit = {
          super.close()
          upstreams.foreach(_.close())
        }

        override def duplicate() = new Output(upstreams.map(_.duplicate()))

      }

      override def forward(input: Input) = new Output(operands.map(_.forward(input)))

    }

  }

  private[deeplearning] type SeqPlaceholder[A <: Placeholder[_, _]] =
    Placeholder[Seq[Placeholder.DataOf[A]], (Int, Placeholder.DeltaOf[A])]

  final class SeqLayerOps[Input <: Tape, ElementData, ElementDelta](
      seqLayer: Layer.Aux[Input, Tape.Aux[Seq[ElementData], (Int, ElementDelta)]]) {

    def apply(i: Int): Layer.Aux[Input, Tape.Aux[ElementData, ElementDelta]] = {
      Get[Input, ElementData, ElementDelta](seqLayer, i)
    }

  }

  /**
    * Implicitly converts any layer to [[SeqLayerOps]], which enables common methods for Seq layers.

    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableSeq._
    * }}}
    */
  implicit def toSeqLayerOps[From, Input <: Tape, SeqData, SeqDelta, ElementData, ElementDelta](from: From)(
      implicit toLayer: ToLayer.Aux[From, Input, SeqData, SeqDelta],
      toSeqLayer: Layer.Aux[Input, Tape.Aux[SeqData, SeqDelta]] <:< Layer.Aux[
        Input,
        Tape.Aux[Seq[ElementData], (Int, ElementDelta)]]
  ): SeqLayerOps[Input, ElementData, ElementDelta] = {
    new SeqLayerOps[Input, ElementData, ElementDelta](toSeqLayer(toLayer(from)))
  }

  implicit def seqToLayer[From, Input0 <: Tape, ElementData, ElementDelta](
      implicit elementToLayer: ToLayer.Aux[From, Input0, ElementData, ElementDelta])
    : ToLayer.Aux[Seq[From], Input0, Seq[ElementData], (Int, ElementDelta)] = {
    new ToLayer[Seq[From], Input0] {
      type OutputData = Seq[ElementData]
      type OutputDelta = (Int, ElementDelta)

      override def apply(layers: Seq[From]): Layer.Aux[Input0, Tape.Aux[Seq[ElementData], (Int, ElementDelta)]] = {
        ToSeq[Input0, ElementData, ElementDelta](layers.map(elementToLayer(_)))
      }
    }
  }

}
