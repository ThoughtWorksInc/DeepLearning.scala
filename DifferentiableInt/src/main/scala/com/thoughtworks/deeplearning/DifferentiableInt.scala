package com.thoughtworks.deeplearning

import cats._
import cats.implicits._
import com.thoughtworks.deeplearning.Layer.Tape
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.Poly.MathMethods
import com.thoughtworks.deeplearning.Poly.MathMethods./
import shapeless.PolyDefns.Case
import shapeless.the

/**
  * A namespace of common operators for Int layers.
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableInt {

  private[deeplearning] type IntPlaceholder = Placeholder[Int, Float]
  private[deeplearning] val IntPlaceholder: IntPlaceholder = new Placeholder

  val Optimizers = DifferentiableDouble.Optimizers

  import Optimizers._

  private[deeplearning] trait IntMonoidTape extends Tape {

    override type Data = Int

    override type Delta = Float

    protected final def monoid = implicitly[Monoid[Delta]]

  }

  object Layers {

    final case class Negative[Input0 <: Tape](operand: Layer.Aux[Input0, IntPlaceholder.Tape])
        extends CumulativeLayer.Unary {

      type CumulativeTape = IntMonoidTape with MonoidTape with UnaryTape

      type Input = Input0

      override protected def rawForward(input0: Input) =
        new {
          override final val input = input0
        } with MonoidTape with IntMonoidTape with UnaryTape {

          val value = -upstream.value

          override protected def rawBackward(delta: Float): Unit = {
            upstream.backward(-delta)
          }

        }

    }

    final case class Plus[Input0 <: Tape](
        operand1: Layer.Aux[Input0, IntPlaceholder.Tape],
        operand2: Layer.Aux[Input0, IntPlaceholder.Tape]
    ) extends CumulativeLayer.Binary {

      type CumulativeTape = IntMonoidTape with MonoidTape with BinaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override final val input = input0
        } with IntMonoidTape with MonoidTape with BinaryTape {

          val value = upstream1.value + upstream2.value

          override protected def rawBackward(delta: Float): Unit = {
            upstream1.backward(delta)
            upstream2.backward(delta)
          }

        }
      }
    }

    final case class Times[Input0 <: Tape](
        operand1: Layer.Aux[Input0, IntPlaceholder.Tape],
        operand2: Layer.Aux[Input0, IntPlaceholder.Tape]
    ) extends CumulativeLayer.Binary {

      type CumulativeTape = IntMonoidTape with MonoidTape with BinaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override final val input = input0
        } with IntMonoidTape with MonoidTape with BinaryTape {

          val value = upstream1.value * upstream2.value

          override protected def rawBackward(delta: Float): Unit = {
            upstream1.backward(delta * upstream1.value)
            upstream2.backward(delta * upstream2.value)
          }

        }
      }
    }

    final case class Reciprocal[Input0 <: Tape](operand: Layer.Aux[Input0, IntPlaceholder.Tape])
        extends CumulativeLayer.Unary {

      type CumulativeTape = IntMonoidTape with MonoidTape with UnaryTape

      type Input = Input0

      override protected def rawForward(input0: Input) =
        new {
          override final val input = input0
        } with MonoidTape with IntMonoidTape with UnaryTape {

          val value = the[Numeric[Int]].one / upstream.value

          override protected def rawBackward(delta: Float): Unit = {
            val a = upstream.value

            upstream.backward(-delta / (a * a))
          }

        }

    }

    final case class Substract[Input0 <: Tape](
        operand1: Layer.Aux[Input0, IntPlaceholder.Tape],
        operand2: Layer.Aux[Input0, IntPlaceholder.Tape]
    ) extends CumulativeLayer.Binary {

      type CumulativeTape = MonoidTape with IntMonoidTape with BinaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override final val input = input0
        } with MonoidTape with IntMonoidTape with BinaryTape {

          val value = upstream1.value - upstream2.value

          override protected def rawBackward(delta: Float): Unit = {
            upstream1.backward(delta)
            upstream2.backward(-delta)
          }

        }
      }
    }

    final case class Weight(var value: Int)(implicit optimizer: Optimizer) extends Layer with IntMonoidTape {
      override type Input = Tape
      override type Output = Tape.Aux[Data, Delta]

      override def isTrainable = true

      override def duplicate() = this

      override def forward(any: Input) = this

      override protected def forceBackward(delta: Delta): Unit = {
        synchronized {
          value = math.rint(optimizer.updateDouble(value, delta)).toInt
        }
      }

      override def close(): Unit = {}

    }
  }

  import com.thoughtworks.deeplearning.DifferentiableInt.Layers._

  implicit final class ScalaIntOps(scalaInt: Int) {
    def toWeight[InputData, InputDelta](
        implicit inputType: Placeholder[InputData, InputDelta],
        optimizer: Optimizer): Layer.Aux[Tape.Aux[InputData, InputDelta], IntPlaceholder.Tape] = {
      Weight(scalaInt)
    }
  }

  implicit def intToLiteral: ToLiteral.Aux[Int, Int, Float] = ToLiteral.fromData

  /**
    * Returns a [[Poly.MathMethods.+.Case]] that accepts two Int [[Layer]]s.
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods.+]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableInt._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputIntLayer: Int @Symbolic)(anotherIntLayer: Int @Symbolic) = {
    *   Poly.MathMethods.+(inputIntLayer,anotherIntLayer)
    * }
    * }}}
    */
  implicit def `Int+Int`[Input <: Tape]: MathMethods.+.Case.Aux[Layer.Aux[Input, IntPlaceholder.Tape],
                                                                Layer.Aux[Input, IntPlaceholder.Tape],
                                                                Layer.Aux[Input, IntPlaceholder.Tape]] = {

    MathMethods.+.at(Plus(_, _))
  }

  /**
    * Returns a [[Poly.MathMethods.-.Case]] that accepts two Int [[Layer]]s.
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods.-]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableInt._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputIntLayer: Int @Symbolic)(anotherIntLayer: Int @Symbolic) = {
    *   Poly.MathMethods.-(inputIntLayer,anotherIntLayer)
    * }
    * }}}
    */
  implicit def `Int-Int`[Input <: Tape]: MathMethods.-.Case.Aux[Layer.Aux[Input, IntPlaceholder.Tape],
                                                                Layer.Aux[Input, IntPlaceholder.Tape],
                                                                Layer.Aux[Input, IntPlaceholder.Tape]] = {

    MathMethods.-.at { (leftLayer, rightLayer) =>
      Plus(leftLayer, Negative(rightLayer))
    }
  }

  /**
    * Returns a [[Poly.MathMethods.*.Case]] that accepts two Int [[Layer]]s.
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods.*]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableInt._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputIntLayer: Int @Symbolic)(anotherIntLayer: Int @Symbolic) = {
    *   Poly.MathMethods.*(inputIntLayer,anotherIntLayer)
    * }
    * }}}
    */
  implicit def `Int*Int`[Input <: Tape]: MathMethods.*.Case.Aux[Layer.Aux[Input, IntPlaceholder.Tape],
                                                                Layer.Aux[Input, IntPlaceholder.Tape],
                                                                Layer.Aux[Input, IntPlaceholder.Tape]] = {

    MathMethods.*.at(Times(_, _))
  }

  /**
    * Returns a [[Poly.MathMethods./.Case]] that accepts two Int [[Layer]]s.
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods./]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableInt._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputIntLayer: Int @Symbolic)(anotherIntLayer: Int @Symbolic) = {
    *   Poly.MathMethods./(inputIntLayer,anotherIntLayer)
    * }
    * }}}
    */
  implicit def `Int/Int`[Input <: Tape]: /.Case.Aux[Layer.Aux[Input, IntPlaceholder.Tape],
                                                    Layer.Aux[Input, IntPlaceholder.Tape],
                                                    Layer.Aux[Input, IntPlaceholder.Tape]] = {
    /.at { (leftLayer, rightLayer) =>
      Times(leftLayer, Reciprocal(rightLayer))
    }
  }

  /**
    * @see [[com.thoughtworks.deeplearning.DifferentiableAny.Trainable Trainable]]
    */
  implicit def intTrainable: Trainable[Int, Float] = new Trainable[Int, Float] {
    def apply(data: Int): Float = data.toFloat
  }

}
