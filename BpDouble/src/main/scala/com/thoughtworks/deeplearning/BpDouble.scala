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

    override type Data = Eval[Double]

    override type Delta = Eval[Double]

    protected final def monoid = implicitly[Monoid[Delta]]

  }

  /** @template */
  type BpDouble = BackPropagationType[Eval[Double], Eval[Double]]

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

    final case class Exp[Input0 <: Batch](operand: Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]])
        extends BufferedLayer.Unary {

      type BufferedBatch = DoubleMonoidBatch with MonoidBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input) =
        new {
          override final val input = input0
        } with MonoidBatch with DoubleMonoidBatch with UnaryBatch {

          val value = upstream.value.map(math.exp).memoize

          override protected def rawBackward(outputDelta: Eval[Double]): Unit = {
            upstream.backward(value.map2(outputDelta)(_ * _).memoize)
          }

        }

    }

    final case class LessThan[Input0 <: Batch](
        operand1: Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]],
        operand2: Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = BooleanMonoidBatch with MonoidBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with BooleanMonoidBatch with MonoidBatch with BinaryBatch {
          override val value = upstream1.value.map2(upstream2.value)(_ < _).memoize

          override protected def rawBackward(delta: Eval[Boolean]): Unit = {
            upstream1.backward(Eval.now(0.0))
            upstream2.backward(Eval.now(0.0))
          }
        }
      }
    }

    final case class Log[Input0 <: Batch](operand: Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]])
        extends BufferedLayer.Unary {

      type BufferedBatch = DoubleMonoidBatch with MonoidBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input) =
        new {
          override final val input = input0
        } with MonoidBatch with DoubleMonoidBatch with UnaryBatch {

          val value = upstream.value.map(math.log).memoize

          override protected def rawBackward(outputDelta: Eval[Double]): Unit = {
            upstream.backward(outputDelta.map2(upstream.value)(_ / _).memoize)
          }

        }

    }

    final case class Negative[Input0 <: Batch](operand: Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]])
        extends BufferedLayer.Unary {

      type BufferedBatch = DoubleMonoidBatch with MonoidBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input) =
        new {
          override final val input = input0
        } with MonoidBatch with DoubleMonoidBatch with UnaryBatch {

          val value = upstream.value.map(-_)

          override protected def rawBackward(delta: Eval[Double]): Unit = {
            upstream.backward(delta.map(-_))
          }

        }

    }

    final case class Plus[Input0 <: Batch](
        operand1: Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]],
        operand2: Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = DoubleMonoidBatch with MonoidBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with DoubleMonoidBatch with MonoidBatch with BinaryBatch {

          val value = upstream1.value.map2(upstream2.value)(_ + _)

          override protected def rawBackward(delta: Eval[Double]): Unit = {
            upstream1.backward(delta)
            upstream2.backward(delta)
          }

        }
      }
    }

    final case class Reciprocal[Input0 <: Batch](operand: Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]])
        extends BufferedLayer.Unary {

      type BufferedBatch = DoubleMonoidBatch with MonoidBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input) =
        new {
          override final val input = input0
        } with MonoidBatch with DoubleMonoidBatch with UnaryBatch {

          val value = upstream.value.map(1.0 / _)

          override protected def rawBackward(delta: Eval[Double]): Unit = {
            val a = upstream.value
            upstream.backward(delta.map2(a) { (outputDeltaValue: Double, aValue: Double) =>
              -outputDeltaValue / (aValue * aValue)
            })
          }

        }

    }

    final case class Substract[Input0 <: Batch](
        operand1: Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]],
        operand2: Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = DoubleMonoidBatch with MonoidBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with DoubleMonoidBatch with MonoidBatch with BinaryBatch {

          val value = upstream1.value.map2(upstream2.value)(_ - _)

          override protected def rawBackward(delta: Eval[Double]): Unit = {
            upstream1.backward(delta)
            upstream2.backward(delta.map(-_))
          }

        }
      }
    }

    final case class Times[Input0 <: Batch](
        operand1: Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]],
        operand2: Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = DoubleMonoidBatch with MonoidBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with DoubleMonoidBatch with MonoidBatch with BinaryBatch {

          override final val value = upstream1.value.map2(upstream2.value)(_ * _)

          override protected def rawBackward(delta: Eval[Double]): Unit = {
            val a = upstream1.value
            val b = upstream2.value
            upstream1.backward(delta.map2(b)(_ * _))
            upstream2.backward(delta.map2(a)(_ * _))
          }

        }
      }
    }

    final case class Weight(var rawValue: Double)(implicit optimizer: Optimizer) extends Layer with DoubleMonoidBatch {
      override type Input = Batch
      override type Output = Batch.Aux[Data, Delta]

      override def addReference() = this

      override def forward(any: Input) = this

      override def backward(delta: Delta): Unit = {
        synchronized {
          rawValue = optimizer.updateDouble(rawValue, delta.value)
        }
      }

      override def value = Eval.now(rawValue)

      override def close(): Unit = {}

    }

  }

  implicit def liftNativeDoubleToLayer[InputData, InputDelta](
      implicit inputType: BackPropagationType[InputData, InputDelta])
    : ToLayer.Aux[Double, Batch.Aux[InputData, InputDelta], Eval[Double], Eval[Double]] =
    new ToLayer[Double, Batch.Aux[InputData, InputDelta]] {
      override type OutputData = Eval[Double]
      override type OutputDelta = Eval[Double]
      override def apply(nativeDouble: Double) = {
        Literal(Eval.now(nativeDouble))
      }
    }

  implicit def `min(Double,Double)`[Input <: Batch]: min.Case.Aux[Layer.Aux[Input, BpDouble#Batch],
                                                                  Layer.Aux[Input, BpDouble#Batch],
                                                                  Layer.Aux[Input, BpDouble#Batch]] = {
    min.at { (leftLayer, rightLayer) =>
      If[Input, BpDouble#Data, BpDouble#Delta](LessThan[Input](leftLayer, rightLayer), leftLayer, rightLayer)
    }
  }
  implicit def `max(Double,Double)`[Input <: Batch]: max.Case.Aux[Layer.Aux[Input, BpDouble#Batch],
                                                                  Layer.Aux[Input, BpDouble#Batch],
                                                                  Layer.Aux[Input, BpDouble#Batch]] = {
    max.at { (leftLayer, rightLayer) =>
      If[Input, BpDouble#Data, BpDouble#Delta](LessThan[Input](leftLayer, rightLayer), rightLayer, leftLayer)
    }
  }

  implicit def `Double-Double`[Input <: Batch]: -.Case.Aux[Layer.Aux[Input, BpDouble#Batch],
                                                           Layer.Aux[Input, BpDouble#Batch],
                                                           Layer.Aux[Input, BpDouble#Batch]] = {
    MathMethods.-.at { (leftLayer, rightLayer) =>
      Plus(leftLayer, Negative(rightLayer))
    }
  }

  implicit def `Double+Double`[Input <: Batch]: +.Case.Aux[Layer.Aux[Input, BpDouble#Batch],
                                                           Layer.Aux[Input, BpDouble#Batch],
                                                           Layer.Aux[Input, BpDouble#Batch]] = {
    MathMethods.+.at { (leftLayer, rightLayer) =>
      Plus(leftLayer, rightLayer)
    }
  }

  implicit def `Double/Double`[Input <: Batch]: /.Case.Aux[Layer.Aux[Input, BpDouble#Batch],
                                                           Layer.Aux[Input, BpDouble#Batch],
                                                           Layer.Aux[Input, BpDouble#Batch]] = {
    /.at { (leftLayer, rightLayer) =>
      Times(leftLayer, Reciprocal(rightLayer))
    }
  }

  implicit def `Double*Double`[Input <: Batch]: *.Case.Aux[Layer.Aux[Input, BpDouble#Batch],
                                                           Layer.Aux[Input, BpDouble#Batch],
                                                           Layer.Aux[Input, BpDouble#Batch]] = {
    *.at(Times(_, _))
  }

  implicit def `log(Double)`[Input <: Batch]
    : log.Case.Aux[Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch]] = {
    log.at(Log(_))
  }

  implicit def `exp(Double)`[Input <: Batch]
    : exp.Case.Aux[Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch]] = {
    exp.at(Exp(_))
  }

  implicit def `abs(Double)`[Input <: Batch]
    : abs.Case.Aux[Layer.Aux[Input, BpDouble#Batch], Layer.Aux[Input, BpDouble#Batch]] = {
    abs.at { operand =>
      If[Input, BpDouble#Data, BpDouble#Delta](LessThan(operand, Literal(Eval.now(0.0))), Negative(operand), operand)
    }
  }

  implicit final class NativeDoubleOps(nativeDouble: Double) {
    def toWeight[InputData, InputDelta](
        implicit inputType: BackPropagationType[InputData, InputDelta],
        optimizer: Optimizer): Layer.Aux[Batch.Aux[InputData, InputDelta], BpDouble#Batch] = {
      Weight(nativeDouble)
    }
  }

  final class DoubleOps[Input <: Batch](differentiable: Layer.Aux[Input, BpDouble#Batch]) {

    def unary_- : Layer.Aux[Input, BpDouble#Batch] = {
      Negative(differentiable)
    }

  }

  implicit def toDoubleOps[From, Input <: Batch](from: From)(
      implicit toLayer: ToLayer.OfType[From, Input, BpDouble]
  ): DoubleOps[Input] = {
    new DoubleOps(toLayer(from))
  }
}
