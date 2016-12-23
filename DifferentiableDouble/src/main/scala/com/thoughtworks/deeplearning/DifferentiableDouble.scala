package com.thoughtworks.deeplearning
import cats.{Eval, Monoid}
import cats.implicits._
import com.thoughtworks.deeplearning.DifferentiableBoolean.BooleanMonoidBatch
import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.Poly.MathMethods._
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import com.thoughtworks.deeplearning.DifferentiableBoolean.Layers.If
import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.Poly.MathMethods
import com.thoughtworks.deeplearning.Lift.Layers.Literal
import com.thoughtworks.deeplearning.DifferentiableDouble.Layers._
import com.thoughtworks.deeplearning.DifferentiableDouble.Optimizers.{LearningRate, Optimizer}
import shapeless.the

import language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableDouble {

  private[deeplearning] trait DoubleMonoidBatch extends Batch {

    override type Data = Double

    override type Delta = Double

    protected final def monoid = implicitly[Monoid[Delta]]

  }

  private[deeplearning] type DoublePlaceholder = Placeholder[Double, Double]

  private[deeplearning] val DoublePlaceholder: DoublePlaceholder = implicitly

  object Optimizers {

    trait Optimizer {
      def updateDouble(oldValue: Double, delta: Double): Double
    }

    trait LearningRate extends Optimizer {

      protected def currentLearningRate(): Double

      override def updateDouble(oldValue: Double, delta: Double): Double = {
        oldValue - delta * currentLearningRate()
      }
    }

    trait L1Regularization extends LearningRate {
      protected def l1Regularization: Double

      override def updateDouble(oldValue: Double, delta: Double): Double = {
        super.updateDouble(oldValue, delta) - math.signum(oldValue) * l1Regularization * currentLearningRate()
      }

    }

    trait L2Regularization extends LearningRate {
      protected def l2Regularization: Double

      override def updateDouble(oldValue: Double, delta: Double): Double = {
        super.updateDouble(oldValue, delta) - l2Regularization * oldValue * currentLearningRate()
      }

    }

  }

  object Layers {

    final case class Exp[Input0 <: Batch](operand: Layer.Aux[Input0, DoublePlaceholder.Batch])
        extends BufferedLayer.Unary {

      type BufferedBatch = DoubleMonoidBatch with MonoidBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input) =
        new {
          override final val input = input0
        } with MonoidBatch with DoubleMonoidBatch with UnaryBatch {

          val value = math.exp(upstream.value)

          override protected def rawBackward(outputDelta: Double): Unit = {
            upstream.backward(value * outputDelta)
          }

        }

    }

    final case class LessThan[Input0 <: Batch](
        operand1: Layer.Aux[Input0, DoublePlaceholder.Batch],
        operand2: Layer.Aux[Input0, DoublePlaceholder.Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = BooleanMonoidBatch with MonoidBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with BooleanMonoidBatch with MonoidBatch with BinaryBatch {
          override val value = upstream1.value < upstream2.value

          override protected def rawBackward(delta: Delta): Unit = {
            // No backward pass
          }
        }
      }
    }

    final case class Log[Input0 <: Batch](operand: Layer.Aux[Input0, DoublePlaceholder.Batch])
        extends BufferedLayer.Unary {

      type BufferedBatch = DoubleMonoidBatch with MonoidBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input) =
        new {
          override final val input = input0
        } with MonoidBatch with DoubleMonoidBatch with UnaryBatch {

          val value = math.log(upstream.value)

          override protected def rawBackward(outputDelta: Double): Unit = {
            upstream.backward(outputDelta / upstream.value)
          }

        }

    }

    final case class Negative[Input0 <: Batch](operand: Layer.Aux[Input0, DoublePlaceholder.Batch])
        extends BufferedLayer.Unary {

      type BufferedBatch = DoubleMonoidBatch with MonoidBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input) =
        new {
          override final val input = input0
        } with MonoidBatch with DoubleMonoidBatch with UnaryBatch {

          val value = -upstream.value

          override protected def rawBackward(delta: Double): Unit = {
            upstream.backward(-delta)
          }

        }

    }

    final case class Plus[Input0 <: Batch](
        operand1: Layer.Aux[Input0, DoublePlaceholder.Batch],
        operand2: Layer.Aux[Input0, DoublePlaceholder.Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = DoubleMonoidBatch with MonoidBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with DoubleMonoidBatch with MonoidBatch with BinaryBatch {

          val value = upstream1.value + upstream2.value

          override protected def rawBackward(delta: Double): Unit = {
            upstream1.backward(delta)
            upstream2.backward(delta)
          }

        }
      }
    }

    final case class Reciprocal[Input0 <: Batch](operand: Layer.Aux[Input0, DoublePlaceholder.Batch])
        extends BufferedLayer.Unary {

      type BufferedBatch = DoubleMonoidBatch with MonoidBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input) =
        new {
          override final val input = input0
        } with MonoidBatch with DoubleMonoidBatch with UnaryBatch {

          val value = 1.0 / upstream.value

          override protected def rawBackward(delta: Double): Unit = {
            val a = upstream.value

            upstream.backward(-delta / (a * a))
          }

        }

    }

    final case class Substract[Input0 <: Batch](
        operand1: Layer.Aux[Input0, DoublePlaceholder.Batch],
        operand2: Layer.Aux[Input0, DoublePlaceholder.Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = DoubleMonoidBatch with MonoidBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with DoubleMonoidBatch with MonoidBatch with BinaryBatch {

          val value = upstream1.value - upstream2.value

          override protected def rawBackward(delta: Double): Unit = {
            upstream1.backward(delta)
            upstream2.backward(-delta)
          }

        }
      }
    }

    final case class Times[Input0 <: Batch](
        operand1: Layer.Aux[Input0, DoublePlaceholder.Batch],
        operand2: Layer.Aux[Input0, DoublePlaceholder.Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = DoubleMonoidBatch with MonoidBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with DoubleMonoidBatch with MonoidBatch with BinaryBatch {

          override final val value = upstream1.value * upstream2.value

          override protected def rawBackward(delta: Double): Unit = {
            val a = upstream1.value
            val b = upstream2.value
            upstream1.backward(delta * b)
            upstream2.backward(delta * a)
          }

        }
      }
    }

    final case class Weight(var value: Double)(implicit optimizer: Optimizer) extends Layer with DoubleMonoidBatch {
      override type Input = Batch
      override type Output = Batch.Aux[Data, Delta]

      override def addReference() = this

      override def forward(any: Input) = this

      override def backward(delta: Delta): Unit = {
        synchronized {
          value = optimizer.updateDouble(value, delta)
        }
      }

      override def close(): Unit = {}

    }

  }

  implicit def liftDouble: Lift.Aux[Double, Double, Double] = Lift.fromData

  implicit def `min(Double,Double)`[Input <: Batch]: min.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Batch],
                                                                  Layer.Aux[Input, DoublePlaceholder.Batch],
                                                                  Layer.Aux[Input, DoublePlaceholder.Batch]] = {
    min.at { (leftLayer, rightLayer) =>
      If[Input, DoublePlaceholder.Data, DoublePlaceholder.Delta](LessThan[Input](leftLayer, rightLayer),
                                                                 leftLayer,
                                                                 rightLayer)
    }
  }

  implicit def `max(Double,Double)`[Input <: Batch]: max.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Batch],
                                                                  Layer.Aux[Input, DoublePlaceholder.Batch],
                                                                  Layer.Aux[Input, DoublePlaceholder.Batch]] = {
    max.at { (leftLayer, rightLayer) =>
      If[Input, DoublePlaceholder.Data, DoublePlaceholder.Delta](LessThan[Input](leftLayer, rightLayer),
                                                                 rightLayer,
                                                                 leftLayer)
    }
  }

  implicit def `Double-Double`[Input <: Batch]: -.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Batch],
                                                           Layer.Aux[Input, DoublePlaceholder.Batch],
                                                           Layer.Aux[Input, DoublePlaceholder.Batch]] = {
    MathMethods.-.at { (leftLayer, rightLayer) =>
      Plus(leftLayer, Negative(rightLayer))
    }
  }

  implicit def `Double+Double`[Input <: Batch]: +.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Batch],
                                                           Layer.Aux[Input, DoublePlaceholder.Batch],
                                                           Layer.Aux[Input, DoublePlaceholder.Batch]] = {
    MathMethods.+.at { (leftLayer, rightLayer) =>
      Plus(leftLayer, rightLayer)
    }
  }

  implicit def `Double/Double`[Input <: Batch]: /.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Batch],
                                                           Layer.Aux[Input, DoublePlaceholder.Batch],
                                                           Layer.Aux[Input, DoublePlaceholder.Batch]] = {
    /.at { (leftLayer, rightLayer) =>
      Times(leftLayer, Reciprocal(rightLayer))
    }
  }

  implicit def `Double*Double`[Input <: Batch]: *.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Batch],
                                                           Layer.Aux[Input, DoublePlaceholder.Batch],
                                                           Layer.Aux[Input, DoublePlaceholder.Batch]] = {
    *.at(Times(_, _))
  }

  implicit def `log(Double)`[Input <: Batch]
    : log.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Batch], Layer.Aux[Input, DoublePlaceholder.Batch]] = {
    log.at(Log(_))
  }

  implicit def `exp(Double)`[Input <: Batch]
    : exp.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Batch], Layer.Aux[Input, DoublePlaceholder.Batch]] = {
    exp.at(Exp(_))
  }

  implicit def `abs(Double)`[Input <: Batch]
    : abs.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Batch], Layer.Aux[Input, DoublePlaceholder.Batch]] = {
    abs.at { operand =>
      If[Input, DoublePlaceholder.Data, DoublePlaceholder.Delta](LessThan(operand, Literal(0.0)),
                                                                 Negative(operand),
                                                                 operand)
    }
  }

  implicit final class NativeDoubleOps(nativeDouble: Double) {
    def toWeight[InputData, InputDelta](
        implicit inputType: Placeholder[InputData, InputDelta],
        optimizer: Optimizer): Layer.Aux[Batch.Aux[InputData, InputDelta], DoublePlaceholder.Batch] = {
      Weight(nativeDouble)
    }
  }

  final class DoubleLayerOps[Input <: Batch](differentiable: Layer.Aux[Input, DoublePlaceholder.Batch]) {

    def unary_- : Layer.Aux[Input, DoublePlaceholder.Batch] = {
      Negative(differentiable)
    }

  }

  implicit def toDoubleLayerOps[From, Input <: Batch](from: From)(
      implicit toLayer: ToLayer.OfType[From, Input, DoublePlaceholder]
  ): DoubleLayerOps[Input] = {
    new DoubleLayerOps(toLayer(from))
  }

//  implicit def doubleToLayer[Input <: Batch]: ToLayer.Aux[Layer.Aux[Input, DoubleBatch], Input, Double, Double] = {
//    Lift.ToLayer.layerToLayer
//  }

}
