package com.thoughtworks.deeplearning

import cats.{Eval, Monoid}
import cats.implicits._
import com.thoughtworks.deeplearning.Layer.{Aux, Tape}
import com.thoughtworks.deeplearning.Symbolic._
import shapeless.Lub

import language.implicitConversions

/**
  * A namespace of common operators for Boolean layers.
  *
  * After importing `DifferentiableBoolean._`, the following methods will be available on Boolean layers.
  *  - [[DifferentiableBoolean.BooleanLayerOps.if if]]
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableBoolean {

  private[deeplearning] trait BooleanMonoidTape extends Tape {

    override type Data = Boolean

    override type Delta = Boolean

    protected final def monoid = new Monoid[Delta] {
      override def empty = false

      override def combine(x: Delta, y: Delta) = x ^ y
    }

  }

  object Layers {

    final case class If[Input0 <: Tape, OutputData0, OutputDelta0](
        condition: Layer.Aux[Input0, BooleanPlaceholder.Tape],
        `then`: Layer.Aux[Input0, Tape.Aux[OutputData0, OutputDelta0]],
        `else`: Layer.Aux[Input0, Tape.Aux[OutputData0, OutputDelta0]])
        extends Layer {
      override type Input = Input0
      override type Output = Tape.Aux[OutputData0, OutputDelta0]

      override def forward(input: Input0) = {
        resource.managed(condition.forward(input)).acquireAndGet { conditionTape =>
          (if (conditionTape.value) `then` else `else`).forward(input)
        }
      }
    }

    final case class Not[Input0 <: Tape](operand: Layer.Aux[Input0, BooleanPlaceholder.Tape])
        extends CumulativeLayer.Unary {

      type CumulativeTape = MonoidTape with BooleanMonoidTape with UnaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {

          override val input = input0

        } with MonoidTape with BooleanMonoidTape with UnaryTape {

          override val value = !upstream.value

          override protected def rawBackward(delta: Boolean): Unit = {
            upstream.backward(!delta)
          }

        }
      }
    }

    final case class Weight[Input0 <: Tape](var value: Boolean) extends Layer with BooleanMonoidTape {
      override type Input = Input0
      override type Output = Weight[Input0]

      override def forward(any: Input) = this

      override protected def forceBackward(delta: Delta): Unit = {
        value ^= delta
      }

      override def close(): Unit = {}

      override def duplicate(): Weight[Input0] = this

      override def isTrainable = true

    }

  }

  import Layers._

  private[deeplearning] type BooleanPlaceholder = Placeholder[Boolean, Boolean]
  private[deeplearning] val BooleanPlaceholder: BooleanPlaceholder = implicitly

  /**
    * A helper that contains methods for all Boolean layers.
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableBoolean._
    * }}}
    */
  final class BooleanLayerOps[Input <: Tape](booleanLayer: Layer.Aux[Input, BooleanPlaceholder.Tape]) {

    def `if`[Then,
             Else,
             ThenOutputData,
             ThenOutputDelta,
             ElseOutputData,
             ElseOutputDelta,
             NN,
             OutputData,
             OutputDelta](`then`: Then)(`else`: Else)(
        implicit thenToLayer: ToLayer.Aux[Then, Input, ThenOutputData, ThenOutputDelta],
        elseToLayer: ToLayer.Aux[Else, Input, ElseOutputData, ElseOutputDelta],
        lub: Lub[Layer.Aux[Input, Tape.Aux[ThenOutputData, ThenOutputDelta]],
                 Layer.Aux[Input, Tape.Aux[ElseOutputData, ElseOutputDelta]],
                 NN],
        commonToLayer: ToLayer.Aux[NN, Input, OutputData, OutputDelta]
    ): Layer.Aux[Input, Tape.Aux[OutputData, OutputDelta]] = {
      If[Input, OutputData, OutputDelta](booleanLayer,
                                         commonToLayer(lub.left(thenToLayer(`then`))),
                                         commonToLayer(lub.right(elseToLayer(`else`))))
    }

  }

  /**
    * Implicitly converts any layer to [[BooleanLayerOps]], which enables common methods for Boolean layers.
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableBoolean._
    * }}}
    */
  implicit def toBooleanLayerOps[From, Input <: Tape](from: From)(
      implicit toLayer: ToLayer.OfPlaceholder[From, Input, BooleanPlaceholder]): BooleanLayerOps[Input] = {
    new BooleanLayerOps[Input](toLayer(from))
  }

  implicit def booleanToLiteral: ToLiteral.Aux[Boolean, Boolean, Boolean] = ToLiteral.fromData

}
