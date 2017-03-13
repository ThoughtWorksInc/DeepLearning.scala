package com.thoughtworks.deeplearning

import cats._
import cats.implicits._
import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.Poly.MathMethods
import com.thoughtworks.deeplearning.Poly.MathMethods./
import shapeless.PolyDefns.Case
import shapeless.the

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableInt {

  private[deeplearning] type IntPlaceholder = Placeholder[Int, Float]
  private[deeplearning] val IntPlaceholder: IntPlaceholder = new Placeholder

  val Optimizers = DifferentiableDouble.Optimizers

  import Optimizers._

  private[deeplearning] trait IntMonoidBatch extends Batch {

    override type Data = Int

    override type Delta = Float

    protected final def monoid = implicitly[Monoid[Delta]]

  }

  object Layers {

    final case class Negative[Input0 <: Batch](operand: Layer.Aux[Input0, IntPlaceholder.Batch])
        extends BufferedLayer.Unary {

      type BufferedBatch = IntMonoidBatch with MonoidBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input) =
        new {
          override final val input = input0
        } with MonoidBatch with IntMonoidBatch with UnaryBatch {

          val value = -upstream.value

          override protected def rawBackward(delta: Float): Unit = {
            upstream.backward(-delta)
          }

        }

    }

    final case class Plus[Input0 <: Batch](
        operand1: Layer.Aux[Input0, IntPlaceholder.Batch],
        operand2: Layer.Aux[Input0, IntPlaceholder.Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = IntMonoidBatch with MonoidBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with IntMonoidBatch with MonoidBatch with BinaryBatch {

          val value = upstream1.value + upstream2.value

          override protected def rawBackward(delta: Float): Unit = {
            upstream1.backward(delta)
            upstream2.backward(delta)
          }

        }
      }
    }

    final case class Times[Input0 <: Batch](
        operand1: Layer.Aux[Input0, IntPlaceholder.Batch],
        operand2: Layer.Aux[Input0, IntPlaceholder.Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = IntMonoidBatch with MonoidBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with IntMonoidBatch with MonoidBatch with BinaryBatch {

          val value = upstream1.value * upstream2.value

          override protected def rawBackward(delta: Float): Unit = {
            upstream1.backward(delta * upstream1.value)
            upstream2.backward(delta * upstream2.value)
          }

        }
      }
    }

    final case class Reciprocal[Input0 <: Batch](operand: Layer.Aux[Input0, IntPlaceholder.Batch])
        extends BufferedLayer.Unary {

      type BufferedBatch = IntMonoidBatch with MonoidBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input) =
        new {
          override final val input = input0
        } with MonoidBatch with IntMonoidBatch with UnaryBatch {

          val value = the[Numeric[Int]].one / upstream.value

          override protected def rawBackward(delta: Float): Unit = {
            val a = upstream.value

            upstream.backward(-delta / (a * a))
          }

        }

    }

    final case class Substract[Input0 <: Batch](
        operand1: Layer.Aux[Input0, IntPlaceholder.Batch],
        operand2: Layer.Aux[Input0, IntPlaceholder.Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = MonoidBatch with IntMonoidBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with MonoidBatch with IntMonoidBatch with BinaryBatch {

          val value = upstream1.value - upstream2.value

          override protected def rawBackward(delta: Float): Unit = {
            upstream1.backward(delta)
            upstream2.backward(-delta)
          }

        }
      }
    }

    final case class Weight(var value: Int)(implicit optimizer: Optimizer) extends Layer with IntMonoidBatch {
      override type Input = Batch
      override type Output = Batch.Aux[Data, Delta]

      override def isTrainable = true

      override def addReference() = this

      override def forward(any: Input) = this

      override protected def forceBackward(delta: Delta): Unit = {
        synchronized {
          value = math.rint(optimizer.updateDouble(value, delta)).toInt
        }
      }

      override def close(): Unit = {}

    }
//    final case class Weight(scalaInt: Int) extends Layer {
//
//    }
  }

  import com.thoughtworks.deeplearning.DifferentiableInt.Layers._

  implicit final class ScalaIntOps(scalaInt: Int) {
    def toWeight[InputData, InputDelta](
        implicit inputType: Placeholder[InputData, InputDelta],
        optimizer: Optimizer): Layer.Aux[Batch.Aux[InputData, InputDelta], IntPlaceholder.Batch] = {
      Weight(scalaInt)
    }
  }

  implicit def intToLiteral: ToLiteral.Aux[Int, Int, Float] = ToLiteral.fromData

  /**
    * Returns a [[Poly.MathMethods.+.Case]] that accepts two Int [[Layer]]s for the polymorphic function [[Poly.MathMethods.+]]
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableInt._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputIntLayer: Int @Symbolic)(anotherIntLayer: Int @Symbolic) = {
    *   Poly.MathMethods.+(inputIntLayer,anotherIntLayer)
    * }
    * }}}
    */
  implicit def `Int+Int`[Input <: Batch]: MathMethods.+.Case.Aux[Layer.Aux[Input, IntPlaceholder.Batch],
                                                                 Layer.Aux[Input, IntPlaceholder.Batch],
                                                                 Layer.Aux[Input, IntPlaceholder.Batch]] = {

    MathMethods.+.at(Plus(_, _))
  }

  /**
    * Returns a [[Poly.MathMethods.-.Case]] that accepts two Int [[Layer]]s for the polymorphic function [[Poly.MathMethods.-]]
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableInt._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputIntLayer: Int @Symbolic)(anotherIntLayer: Int @Symbolic) = {
    *   Poly.MathMethods.-(inputIntLayer,anotherIntLayer)
    * }
    * }}}
    */
  implicit def `Int-Int`[Input <: Batch]: MathMethods.-.Case.Aux[Layer.Aux[Input, IntPlaceholder.Batch],
                                                                 Layer.Aux[Input, IntPlaceholder.Batch],
                                                                 Layer.Aux[Input, IntPlaceholder.Batch]] = {

    MathMethods.-.at { (leftLayer, rightLayer) =>
      Plus(leftLayer, Negative(rightLayer))
    }
  }

  /**
    * Returns a [[Poly.MathMethods.*.Case]] that accepts two Int [[Layer]]s for the polymorphic function [[Poly.MathMethods.*]]
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableInt._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputIntLayer: Int @Symbolic)(anotherIntLayer: Int @Symbolic) = {
    *   Poly.MathMethods.*(inputIntLayer,anotherIntLayer)
    * }
    * }}}
    */
  implicit def `Int*Int`[Input <: Batch]: MathMethods.*.Case.Aux[Layer.Aux[Input, IntPlaceholder.Batch],
                                                                 Layer.Aux[Input, IntPlaceholder.Batch],
                                                                 Layer.Aux[Input, IntPlaceholder.Batch]] = {

    MathMethods.*.at(Times(_, _))
  }

  /**
    * Returns a [[Poly.MathMethods./.Case]] that accepts two Int [[Layer]]s for the polymorphic function [[Poly.MathMethods./]]
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableInt._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputIntLayer: Int @Symbolic)(anotherIntLayer: Int @Symbolic) = {
    *   Poly.MathMethods./(inputIntLayer,anotherIntLayer)
    * }
    * }}}
    */
  implicit def `Int/Int`[Input <: Batch]: /.Case.Aux[Layer.Aux[Input, IntPlaceholder.Batch],
                                                     Layer.Aux[Input, IntPlaceholder.Batch],
                                                     Layer.Aux[Input, IntPlaceholder.Batch]] = {
    /.at { (leftLayer, rightLayer) =>
      Times(leftLayer, Reciprocal(rightLayer))
    }
  }

  implicit def intTrainable: Trainable[Int, Float] = new Trainable[Int, Float] {
    def apply(data: Int): Float = data.toFloat
  }

}
