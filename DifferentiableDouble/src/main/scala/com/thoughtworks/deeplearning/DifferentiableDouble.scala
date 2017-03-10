package com.thoughtworks.deeplearning
import cats.{Eval, Monoid}
import cats.implicits._
import com.thoughtworks.deeplearning.DifferentiableAny.Trainable
import com.thoughtworks.deeplearning.DifferentiableBoolean.BooleanMonoidBatch
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.Poly.MathMethods._
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import com.thoughtworks.deeplearning.DifferentiableBoolean.Layers.If
import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.Poly.MathMethods
import com.thoughtworks.deeplearning.Symbolic.Layers.Literal
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

  /**
    * Optimizers of Double
    */
  object Optimizers {

    trait Optimizer {
      def updateDouble(oldValue: Double, delta: Double): Double
    }

    trait LearningRate extends Optimizer {

      protected def currentLearningRate(): Double

      /**
        * Update double use learning rate
        * @param oldValue double value before update
        * @param delta delta
        */
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

  import Optimizers._

  object Layers {

    final case class Exp[Input0 <: Batch](operand: Layer.Aux[Input0, DoublePlaceholder.Batch])
        extends BufferedLayer.Unary {

      type BufferedBatch = DoubleMonoidBatch with MonoidBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input) =
        new {
          override final val input = input0
        } with MonoidBatch with DoubleMonoidBatch with UnaryBatch {

          val value: Double = math.exp(upstream.value).toDouble

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

          val value = math.log(upstream.value).toDouble

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

          val value = the[Numeric[Double]].one / upstream.value

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

    object Weight {

      def apply(value: Double)(implicit optimizerFactory: OptimizerFactory) = new Weight(value) {
        override protected val optimizer = optimizerFactory.doubleOptimizer(this)
      }

    }

    abstract case class Weight(var value: Double) extends Layer with DoubleMonoidBatch {
      override type Input = Batch
      override type Output = Batch.Aux[Data, Delta]

      override final def isTrainable = true

      protected def optimizer: Optimizer

      override final def addReference() = this

      override final def forward(any: Input) = this

      override protected final def forceBackward(delta: Delta): Unit = {
        synchronized {
          value = optimizer.updateDouble(value, delta)
        }
      }

      override final def close(): Unit = {}

    }

  }

  import Layers._

  object OptimizerFactory {
    implicit def shared(implicit optimizer: Optimizer): OptimizerFactory = new OptimizerFactory {
      override def doubleOptimizer(weight: Weight) = optimizer
    }
  }

  trait OptimizerFactory {
    def doubleOptimizer(weight: Weight): Optimizer
  }
  implicit def liftDouble: ToLiteral.Aux[Double, Double, Double] = ToLiteral.fromData

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
      If[Input, DoublePlaceholder.Data, DoublePlaceholder.Delta](LessThan(operand, Literal(the[Numeric[Double]].zero)),
                                                                 Negative(operand),
                                                                 operand)
    }
  }

  implicit final class NativeDoubleOps(nativeDouble: Double) {
    def toWeight[InputData, InputDelta](
        implicit inputType: Placeholder[InputData, InputDelta],
        optimizerFactory: OptimizerFactory): Layer.Aux[Batch.Aux[InputData, InputDelta], DoublePlaceholder.Batch] = {
      Weight(nativeDouble)
    }
  }

  final class DoubleLayerOps[Input <: Batch](differentiable: Layer.Aux[Input, DoublePlaceholder.Batch]) {

    def unary_- : Layer.Aux[Input, DoublePlaceholder.Batch] = {
      Negative(differentiable)
    }

  }

  /**
    * Implicit conversions for all Double layers.
    * {{{
    * import com.thoughtworks.deeplearning.DifferentiableDouble._
    * }}}
    */
  implicit def toDoubleLayerOps[From, Input <: Batch](from: From)(
      implicit toLayer: ToLayer.OfPlaceholder[From, Input, DoublePlaceholder]
  ): DoubleLayerOps[Input] = {
    new DoubleLayerOps(toLayer(from))
  }

  implicit def doubleTrainable: Trainable[Double, Double] = new Trainable[Double, Double] {
    def apply(data: Double): Double = the[Numeric[Double]].one
  }

}
