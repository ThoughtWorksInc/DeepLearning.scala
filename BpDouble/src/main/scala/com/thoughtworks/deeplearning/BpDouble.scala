package com.thoughtworks.deeplearning
import cats.{Eval, Monoid}
import cats.implicits._
import com.thoughtworks.deeplearning.BpBoolean.BooleanMonoidBatch
import com.thoughtworks.deeplearning.Conversion._
import com.thoughtworks.deeplearning.Poly.MathMethods._
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import com.thoughtworks.deeplearning.BpBoolean.Layers.If
import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.Poly.MathMethods
import com.thoughtworks.deeplearning.Conversion.Layers.Literal
import com.thoughtworks.deeplearning.BpDouble.Layers._
import com.thoughtworks.deeplearning.BpDouble.Optimizers.{LearningRate, Optimizer}

import language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object BpDouble {

  private[deeplearning] trait DoubleMonoidBatch extends Batch {

    override type Data = Double

    override type Delta = Double

    protected final def monoid = implicitly[Monoid[Delta]]

  }

  /** @template */
  type DoubleBackProgationType = BackPropagationType[Double, Double]

  private[deeplearning] val DoubleBackProgationType = BackPropagationType[Double, Double]

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

    final case class Exp[Input0 <: Batch](operand: Layer.Aux[Input0, DoubleBackProgationType.Batch])
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
                                                operand1: Layer.Aux[Input0, DoubleBackProgationType.Batch],
                                                operand2: Layer.Aux[Input0, DoubleBackProgationType.Batch]
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

    final case class Log[Input0 <: Batch](operand: Layer.Aux[Input0, DoubleBackProgationType.Batch])
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

    final case class Negative[Input0 <: Batch](operand: Layer.Aux[Input0, DoubleBackProgationType.Batch])
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
                                            operand1: Layer.Aux[Input0, DoubleBackProgationType.Batch],
                                            operand2: Layer.Aux[Input0, DoubleBackProgationType.Batch]
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

    final case class Reciprocal[Input0 <: Batch](operand: Layer.Aux[Input0, DoubleBackProgationType.Batch])
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
                                                 operand1: Layer.Aux[Input0, DoubleBackProgationType.Batch],
                                                 operand2: Layer.Aux[Input0, DoubleBackProgationType.Batch]
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
                                             operand1: Layer.Aux[Input0, DoubleBackProgationType.Batch],
                                             operand2: Layer.Aux[Input0, DoubleBackProgationType.Batch]
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

  implicit def liftNativeDoubleToLiteral: ToLiteral.Aux[Double, Double, Double] =
    new ToLiteral[Double] {
      override type Data = Double
      override type Delta = Double
      override def apply(nativeDouble: Double) = {
        Literal(nativeDouble)
      }
    }

  implicit def `min(Double,Double)`[Input <: Batch]: min.Case.Aux[Layer.Aux[Input, DoubleBackProgationType.Batch],
                                                                  Layer.Aux[Input, DoubleBackProgationType.Batch],
                                                                  Layer.Aux[Input, DoubleBackProgationType.Batch]] = {
    min.at { (leftLayer, rightLayer) =>
      If[Input, DoubleBackProgationType.Data, DoubleBackProgationType.Delta](LessThan[Input](leftLayer, rightLayer), leftLayer, rightLayer)
    }
  }

  implicit def `max(Double,Double)`[Input <: Batch]: max.Case.Aux[Layer.Aux[Input, DoubleBackProgationType.Batch],
                                                                  Layer.Aux[Input, DoubleBackProgationType.Batch],
                                                                  Layer.Aux[Input, DoubleBackProgationType.Batch]] = {
    max.at { (leftLayer, rightLayer) =>
      If[Input, DoubleBackProgationType.Data, DoubleBackProgationType.Delta](LessThan[Input](leftLayer, rightLayer), rightLayer, leftLayer)
    }
  }

  implicit def `Double-Double`[Input <: Batch]: -.Case.Aux[Layer.Aux[Input, DoubleBackProgationType.Batch],
                                                           Layer.Aux[Input, DoubleBackProgationType.Batch],
                                                           Layer.Aux[Input, DoubleBackProgationType.Batch]] = {
    MathMethods.-.at { (leftLayer, rightLayer) =>
      Plus(leftLayer, Negative(rightLayer))
    }
  }

  implicit def `Double+Double`[Input <: Batch]: +.Case.Aux[Layer.Aux[Input, DoubleBackProgationType.Batch],
                                                           Layer.Aux[Input, DoubleBackProgationType.Batch],
                                                           Layer.Aux[Input, DoubleBackProgationType.Batch]] = {
    MathMethods.+.at { (leftLayer, rightLayer) =>
      Plus(leftLayer, rightLayer)
    }
  }

  implicit def `Double/Double`[Input <: Batch]: /.Case.Aux[Layer.Aux[Input, DoubleBackProgationType.Batch],
                                                           Layer.Aux[Input, DoubleBackProgationType.Batch],
                                                           Layer.Aux[Input, DoubleBackProgationType.Batch]] = {
    /.at { (leftLayer, rightLayer) =>
      Times(leftLayer, Reciprocal(rightLayer))
    }
  }

  implicit def `Double*Double`[Input <: Batch]: *.Case.Aux[Layer.Aux[Input, DoubleBackProgationType.Batch],
                                                           Layer.Aux[Input, DoubleBackProgationType.Batch],
                                                           Layer.Aux[Input, DoubleBackProgationType.Batch]] = {
    *.at(Times(_, _))
  }

  implicit def `log(Double)`[Input <: Batch]
    : log.Case.Aux[Layer.Aux[Input, DoubleBackProgationType.Batch], Layer.Aux[Input, DoubleBackProgationType.Batch]] = {
    log.at(Log(_))
  }

  implicit def `exp(Double)`[Input <: Batch]
    : exp.Case.Aux[Layer.Aux[Input, DoubleBackProgationType.Batch], Layer.Aux[Input, DoubleBackProgationType.Batch]] = {
    exp.at(Exp(_))
  }

  implicit def `abs(Double)`[Input <: Batch]
    : abs.Case.Aux[Layer.Aux[Input, DoubleBackProgationType.Batch], Layer.Aux[Input, DoubleBackProgationType.Batch]] = {
    abs.at { operand =>
      If[Input, DoubleBackProgationType.Data, DoubleBackProgationType.Delta](LessThan(operand, Literal(0.0)), Negative(operand), operand)
    }
  }

  implicit final class NativeDoubleOps(nativeDouble: Double) {
    def toWeight[InputData, InputDelta](
        implicit inputType: BackPropagationType[InputData, InputDelta],
        optimizer: Optimizer): Layer.Aux[Batch.Aux[InputData, InputDelta], DoubleBackProgationType.Batch] = {
      Weight(nativeDouble)
    }
  }

  final class DoubleLayerOps[Input <: Batch](differentiable: Layer.Aux[Input, DoubleBackProgationType.Batch]) {

    def unary_- : Layer.Aux[Input, DoubleBackProgationType.Batch] = {
      Negative(differentiable)
    }

  }

  implicit def toDoubleLayerOps[From, Input <: Batch](from: From)(
      implicit toLayer: ToLayer.OfType[From, Input, DoubleBackProgationType]
  ): DoubleLayerOps[Input] = {
    new DoubleLayerOps(toLayer(from))
  }

//  implicit def doubleToLayer[Input <: Batch]: ToLayer.Aux[Layer.Aux[Input, DoubleBatch], Input, Double, Double] = {
//    Conversion.ToLayer.layerToLayer
//  }

}
