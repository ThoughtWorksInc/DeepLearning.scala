package com.thoughtworks.deeplearning
import cats.{Eval, Monoid}
import cats.implicits._
import com.thoughtworks.deeplearning.DifferentiableAny.Trainable
import com.thoughtworks.deeplearning.DifferentiableBoolean.BooleanMonoidTape
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.Poly.MathMethods._
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import com.thoughtworks.deeplearning.DifferentiableBoolean.Layers.If
import com.thoughtworks.deeplearning.Layer.Tape
import com.thoughtworks.deeplearning.Poly.MathMethods
import com.thoughtworks.deeplearning.Symbolic.Layers.Literal
import shapeless.the

import language.implicitConversions

/**
  * A namespace of common operators for Double layers.
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableDouble {

  private[deeplearning] trait DoubleMonoidTape extends Tape {

    override type Data = Double

    override type Delta = Double

    protected final def monoid = implicitly[Monoid[Delta]]

  }

  private[deeplearning] type DoublePlaceholder = Placeholder[Double, Double]

  private[deeplearning] val DoublePlaceholder: DoublePlaceholder = implicitly

  /**
    * Optimizers of Double.
    *
    * @example{{{
    * implicit val optimizerFactory = new DifferentiableDouble.OptimizerFactory {
    *   override def doubleOptimizer(weight: Weight): Optimizer = {
    *     new LearningRate with L2Regularization {
    *
    *       var learningRate = 0.00003
    *
    *       override protected def l2Regularization: Double = 0.003
    *
    *       override protected def currentLearningRate(): Double = {
    *       learningRate * 0.75
    *       learningRate
    *      }
    *    }
    *  }
    * }
    * }}}
    */
  object Optimizers {

    trait Optimizer {
      def currentDelta(oldValue: Double, delta: Double): Double = delta

      final def updateDouble(oldValue: Double, delta: Double): Double = {
        oldValue - currentDelta(oldValue, delta)
      }
    }

    trait LearningRate extends Optimizer {

      protected def currentLearningRate(): Double

      override def currentDelta(oldValue: Double, delta: Double): Double = delta * currentLearningRate()
    }

    trait L1Regularization extends Optimizer {
      protected def l1Regularization: Double

      override def currentDelta(oldValue: Double, delta: Double): Double = {
        super.currentDelta(oldValue, delta + math.signum(oldValue) * l1Regularization)
      }
    }

    trait L2Regularization extends Optimizer {
      protected def l2Regularization: Double

      override def currentDelta(oldValue: Double, delta: Double): Double = {
        super.currentDelta(oldValue, delta + oldValue * l2Regularization)
      }
    }

  }

  import Optimizers._

  object Layers {

    final case class Exp[Input0 <: Tape](operand: Layer.Aux[Input0, DoublePlaceholder.Tape])
        extends CumulativeLayer.Unary {

      type CumulativeTape = DoubleMonoidTape with MonoidTape with UnaryTape

      type Input = Input0

      override protected def rawForward(input0: Input) =
        new {
          override final val input = input0
        } with MonoidTape with DoubleMonoidTape with UnaryTape {

          val value: Double = math.exp(upstream.value).toDouble

          override protected def rawBackward(outputDelta: Double): Unit = {
            upstream.backward(value * outputDelta)
          }

        }

    }

    final case class LessThan[Input0 <: Tape](
        operand1: Layer.Aux[Input0, DoublePlaceholder.Tape],
        operand2: Layer.Aux[Input0, DoublePlaceholder.Tape]
    ) extends CumulativeLayer.Binary {

      type CumulativeTape = BooleanMonoidTape with MonoidTape with BinaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override final val input = input0
        } with BooleanMonoidTape with MonoidTape with BinaryTape {
          override val value = upstream1.value < upstream2.value

          override protected def rawBackward(delta: Delta): Unit = {
            // No backward pass
          }
        }
      }
    }

    final case class Log[Input0 <: Tape](operand: Layer.Aux[Input0, DoublePlaceholder.Tape])
        extends CumulativeLayer.Unary {

      type CumulativeTape = DoubleMonoidTape with MonoidTape with UnaryTape

      type Input = Input0

      override protected def rawForward(input0: Input) =
        new {
          override final val input = input0
        } with MonoidTape with DoubleMonoidTape with UnaryTape {

          val value = math.log(upstream.value).toDouble

          override protected def rawBackward(outputDelta: Double): Unit = {
            upstream.backward(outputDelta / upstream.value)
          }

        }

    }

    final case class Negative[Input0 <: Tape](operand: Layer.Aux[Input0, DoublePlaceholder.Tape])
        extends CumulativeLayer.Unary {

      type CumulativeTape = DoubleMonoidTape with MonoidTape with UnaryTape

      type Input = Input0

      override protected def rawForward(input0: Input) =
        new {
          override final val input = input0
        } with MonoidTape with DoubleMonoidTape with UnaryTape {

          val value = -upstream.value

          override protected def rawBackward(delta: Double): Unit = {
            upstream.backward(-delta)
          }

        }

    }

    final case class Plus[Input0 <: Tape](
        operand1: Layer.Aux[Input0, DoublePlaceholder.Tape],
        operand2: Layer.Aux[Input0, DoublePlaceholder.Tape]
    ) extends CumulativeLayer.Binary {

      type CumulativeTape = DoubleMonoidTape with MonoidTape with BinaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override final val input = input0
        } with DoubleMonoidTape with MonoidTape with BinaryTape {

          val value = upstream1.value + upstream2.value

          override protected def rawBackward(delta: Double): Unit = {
            upstream1.backward(delta)
            upstream2.backward(delta)
          }

        }
      }
    }

    final case class Reciprocal[Input0 <: Tape](operand: Layer.Aux[Input0, DoublePlaceholder.Tape])
        extends CumulativeLayer.Unary {

      type CumulativeTape = DoubleMonoidTape with MonoidTape with UnaryTape

      type Input = Input0

      override protected def rawForward(input0: Input) =
        new {
          override final val input = input0
        } with MonoidTape with DoubleMonoidTape with UnaryTape {

          val value = the[Numeric[Double]].one / upstream.value

          override protected def rawBackward(delta: Double): Unit = {
            val a = upstream.value

            upstream.backward(-delta / (a * a))
          }

        }

    }

    final case class Substract[Input0 <: Tape](
        operand1: Layer.Aux[Input0, DoublePlaceholder.Tape],
        operand2: Layer.Aux[Input0, DoublePlaceholder.Tape]
    ) extends CumulativeLayer.Binary {

      type CumulativeTape = DoubleMonoidTape with MonoidTape with BinaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override final val input = input0
        } with DoubleMonoidTape with MonoidTape with BinaryTape {

          val value = upstream1.value - upstream2.value

          override protected def rawBackward(delta: Double): Unit = {
            upstream1.backward(delta)
            upstream2.backward(-delta)
          }

        }
      }
    }

    final case class Times[Input0 <: Tape](
        operand1: Layer.Aux[Input0, DoublePlaceholder.Tape],
        operand2: Layer.Aux[Input0, DoublePlaceholder.Tape]
    ) extends CumulativeLayer.Binary {

      type CumulativeTape = DoubleMonoidTape with MonoidTape with BinaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override final val input = input0
        } with DoubleMonoidTape with MonoidTape with BinaryTape {

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

    abstract case class Weight(var value: Double) extends Layer with DoubleMonoidTape {
      override type Input = Tape
      override type Output = Tape.Aux[Data, Delta]

      override final def isTrainable = true

      protected def optimizer: Optimizer

      override final def duplicate() = this

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

  implicit def doubleToLiteral: ToLiteral.Aux[Double, Double, Double] = ToLiteral.fromData

  /**
    * Returns a [[Poly.MathFunctions.min.Case  Case]] that accepts two Double [[Layer]]s for the polymorphic function [[Poly.MathFunctions.min min]]
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableDouble._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputDoubleLayer: Double @Symbolic)(anotherDoubleLayer: Double @Symbolic) = {
    *   Poly.MathFunctions.min(inputDoubleLayer,anotherDoubleLayer)
    * }
    * }}}
    */
  implicit def `min(Double,Double)`[Input <: Tape]: min.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Tape],
                                                                 Layer.Aux[Input, DoublePlaceholder.Tape],
                                                                 Layer.Aux[Input, DoublePlaceholder.Tape]] = {
    min.at { (leftLayer, rightLayer) =>
      If[Input, DoublePlaceholder.Data, DoublePlaceholder.Delta](LessThan[Input](leftLayer, rightLayer),
                                                                 leftLayer,
                                                                 rightLayer)
    }
  }

  /**
    * Returns a [[Poly.MathFunctions.max.Case Case]] that accepts two Double [[Layer]]s for the polymorphic function [[Poly.MathFunctions.max max]]
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableDouble._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputDoubleLayer: Double @Symbolic)(anotherDoubleLayer: Double @Symbolic) = {
    *   Poly.MathFunctions.max(inputDoubleLayer,anotherDoubleLayer)
    * }
    * }}}
    */
  implicit def `max(Double,Double)`[Input <: Tape]: max.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Tape],
                                                                 Layer.Aux[Input, DoublePlaceholder.Tape],
                                                                 Layer.Aux[Input, DoublePlaceholder.Tape]] = {
    max.at { (leftLayer, rightLayer) =>
      If[Input, DoublePlaceholder.Data, DoublePlaceholder.Delta](LessThan[Input](leftLayer, rightLayer),
                                                                 rightLayer,
                                                                 leftLayer)
    }
  }

  /**
    * Returns a [[Poly.MathMethods.-.Case Case]] that accepts two Double [[Layer]]s.
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods.- -]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableDouble._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputDoubleLayer: Double @Symbolic)(anotherDoubleLayer: Double @Symbolic) = {
    *   Poly.MathMethods.-(inputDoubleLayer,anotherDoubleLayer)
    * }
    * }}}
    */
  implicit def `Double-Double`[Input <: Tape]: -.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Tape],
                                                          Layer.Aux[Input, DoublePlaceholder.Tape],
                                                          Layer.Aux[Input, DoublePlaceholder.Tape]] = {
    MathMethods.-.at { (leftLayer, rightLayer) =>
      Plus(leftLayer, Negative(rightLayer))
    }
  }

  /**
    * Returns a [[Poly.MathMethods.+.Case Case]] that accepts two Double [[Layer]]s.
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods.+ +]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableDouble._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputDoubleLayer: Double @Symbolic)(anotherDoubleLayer: Double @Symbolic) = {
    *   Poly.MathMethods.+(inputDoubleLayer,anotherDoubleLayer)
    * }
    * }}}
    */
  implicit def `Double+Double`[Input <: Tape]: +.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Tape],
                                                          Layer.Aux[Input, DoublePlaceholder.Tape],
                                                          Layer.Aux[Input, DoublePlaceholder.Tape]] = {
    MathMethods.+.at { (leftLayer, rightLayer) =>
      Plus(leftLayer, rightLayer)
    }
  }

  /**
    * Returns a [[Poly.MathMethods./.Case Case]] that accepts two Double [[Layer]]s.
    *
    * The returned `Case` is used by the  polymorphic function [[Poly.MathMethods./ /]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableDouble._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputDoubleLayer: Double @Symbolic)(anotherDoubleLayer: Double @Symbolic) = {
    *   Poly.MathMethods./(inputDoubleLayer,anotherDoubleLayer)
    * }
    * }}}
    */
  implicit def `Double/Double`[Input <: Tape]: /.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Tape],
                                                          Layer.Aux[Input, DoublePlaceholder.Tape],
                                                          Layer.Aux[Input, DoublePlaceholder.Tape]] = {
    /.at { (leftLayer, rightLayer) =>
      Times(leftLayer, Reciprocal(rightLayer))
    }
  }

  /**
    * Returns a [[Poly.MathMethods.*.Case Case]] that accepts two Double [[Layer]]s.
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods.* *]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableDouble._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputDoubleLayer: Double @Symbolic)(anotherDoubleLayer: Double @Symbolic) = {
    *   inputDoubleLayer * anotherDoubleLayer
    * }
    * }}}
    */
  implicit def `Double*Double`[Input <: Tape]: *.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Tape],
                                                          Layer.Aux[Input, DoublePlaceholder.Tape],
                                                          Layer.Aux[Input, DoublePlaceholder.Tape]] = {
    *.at(Times(_, _))
  }

  /**
    * Returns a [[Poly.MathFunctions.log.Case Case]] that accepts Double [[Layer]] for the polymorphic function [[Poly.MathFunctions.log log]]
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableDouble._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputDoubleLayer: Double @Symbolic) = {
    *   Poly.MathFunctions.log(inputDoubleLayer)
    * }
    * }}}
    */
  implicit def `log(Double)`[Input <: Tape]
    : log.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Tape], Layer.Aux[Input, DoublePlaceholder.Tape]] = {
    log.at(Log(_))
  }

  /**
    * Returns a [[Poly.MathFunctions.exp.Case Case]] that accepts Double [[Layer]] for the polymorphic function [[Poly.MathFunctions.exp exp]]
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableDouble._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputDoubleLayer: Double @Symbolic) = {
    *   Poly.MathFunctions.exp(inputDoubleLayer)
    * }
    * }}}
    */
  implicit def `exp(Double)`[Input <: Tape]
    : exp.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Tape], Layer.Aux[Input, DoublePlaceholder.Tape]] = {
    exp.at(Exp(_))
  }

  /**
    * Returns a [[Poly.MathFunctions.abs.Case Case]] that accepts Double [[Layer]] for the polymorphic function [[Poly.MathFunctions.abs abs]]
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableDouble._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputDoubleLayer: Double @Symbolic) = {
    *   Poly.MathFunctions.abs(inputDoubleLayer)
    * }
    * }}}
    */
  implicit def `abs(Double)`[Input <: Tape]
    : abs.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Tape], Layer.Aux[Input, DoublePlaceholder.Tape]] = {
    abs.at { operand =>
      If[Input, DoublePlaceholder.Data, DoublePlaceholder.Delta](LessThan(operand, Literal(the[Numeric[Double]].zero)),
                                                                 Negative(operand),
                                                                 operand)
    }
  }

  implicit final class NativeDoubleOps(nativeDouble: Double) {
    def toWeight[InputData, InputDelta](
        implicit inputType: Placeholder[InputData, InputDelta],
        optimizerFactory: OptimizerFactory): Layer.Aux[Tape.Aux[InputData, InputDelta], DoublePlaceholder.Tape] = {
      Weight(nativeDouble)
    }
  }

  final class DoubleLayerOps[Input <: Tape](differentiable: Layer.Aux[Input, DoublePlaceholder.Tape]) {

    /**
      * Opposite number
      */
    def unary_- : Layer.Aux[Input, DoublePlaceholder.Tape] = {
      Negative(differentiable)
    }

  }

  /**
    * Implicitly converts any layer to [[DoubleLayerOps]], which enables common methods for Double layers.
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableDouble._
    * }}}
    */
  implicit def toDoubleLayerOps[From, Input <: Tape](from: From)(
      implicit toLayer: ToLayer.OfPlaceholder[From, Input, DoublePlaceholder]
  ): DoubleLayerOps[Input] = {
    new DoubleLayerOps(toLayer(from))
  }

  /**
    * @see [[com.thoughtworks.deeplearning.DifferentiableAny.Trainable Trainable]]
    */
  implicit def doubleTrainable: Trainable[Double, Double] = new Trainable[Double, Double] {
    def apply(data: Double): Double = the[Numeric[Double]].one
  }

}
