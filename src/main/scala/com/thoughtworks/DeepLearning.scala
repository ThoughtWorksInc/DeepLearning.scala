package com.thoughtworks

import cats.{Applicative, Eval, Monoid}

import scala.language.existentials
import scala.language.implicitConversions
import scala.language.higherKinds
import cats.implicits._
import com.thoughtworks.DeepLearning.DifferentiableFunction.Array2DFunction.Aux
import com.thoughtworks.DeepLearning.DifferentiableFunction.DoubleFunction.Aux
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

object Dsl {

  trait DoubleApi {
    type Companion[_]
    type Array2D <: Array2DApi.Aux[Companion, Array2D, Double, Boolean]
    type Double >: this.type <: DoubleApi.Aux[Companion, Array2D, Double, Boolean]
    type Boolean <: BooleanApi.Aux[Companion, Boolean]

    def unary_- : Double

    def -(rightHandSide: Double): Double = {
      this + -rightHandSide
    }

    def -(rightHandSide: Array2D): Array2D = {
      this + -rightHandSide
    }

    def +(rightHandSide: Double): Double

    def +(rightHandSide: Array2D): Array2D = {
      rightHandSide + (this: Double)
    }

    def /(rightHandSide: Double): Double

    def /(rightHandSide: Array2D): Array2D

    def *(rightHandSide: Double): Double

    def *(rightHandSide: Array2D): Array2D = {
      rightHandSide * (this: Double)
    }

    def <(rightHandSide: Double): Boolean

    def >=(rightHandSide: Double): Boolean = {
      !(rightHandSide < (this: Double))
    }

  }

  object DoubleApi {
    type Aux[Companion0[_], Array2D0, Double0, Boolean0] = DoubleApi {
      type Companion[A] = Companion0[A]
      type Array2D = Array2D0
      type Double = Double0
      type Boolean = Boolean0
    }
  }

  trait BooleanApi {
    type Companion[_]
    type Boolean >: this.type <: BooleanApi.Aux[Companion, Boolean]


    def unary_! : Boolean

    def `if`[A: Companion](`then`: A)(`else`: A): A
  }

  object BooleanApi {
    type Aux[Companion0[_], Boolean0] = BooleanApi {
      type Companion[A] = Companion0[A]
      type Boolean = Boolean0
    }
  }

  trait Array2DApi {
    type Double <: DoubleApi.Aux[Companion, Array2D, Double, Boolean]
    type Array2D <: Array2DApi.Aux[Companion, Array2D, Double, Boolean]
    type Boolean <: BooleanApi
    type Companion[_]

    def dot(rightHandSide: Array2D): Array2D

    def +(rightHandSide: Array2D): Array2D

    def +(rightHandSide: Double): Array2D

    def /(rightHandSide: Array2D): Array2D

    def /(rightHandSide: Double): Array2D

    def *(rightHandSide: Array2D): Array2D

    def *(rightHandSide: Double): Array2D

    def -(rightHandSide: Array2D): Array2D = {
      this + -rightHandSide
    }

    def -(rightHandSide: Double): Array2D = {
      this + -rightHandSide
    }

    def unary_- : Array2D

  }

  object Array2DApi {
    type Aux[Companion0[_], Array2D0, Double0, Boolean0] = Array2DApi {
      type Companion[A] = Companion0[A]
      type Array2D = Array2D0
      type Double = Double0
      type Boolean = Boolean0
    }
  }

  object Lifter {
    type Aux[LiftFrom0, LiftTo0] = Lifter {
      type LiftFrom = LiftFrom0
      type LiftTo = LiftTo0
    }
  }

  trait Lifter {
    type LiftFrom
    type LiftTo

    def weight(initialValue: LiftFrom): LiftTo

    def apply(value: LiftFrom): LiftTo
  }

}

trait Dsl {

  import Dsl._

  type Companion[_]

  type Any
  implicit val Any: Companion[Any]

  type Boolean <: Any with BooleanApi.Aux[Companion, Boolean]
  implicit val Boolean: Companion[Boolean]

  type Double <: Any with DoubleApi.Aux[Companion, Array2D, Double, Boolean]
  implicit val Double: Companion[Double] with Lifter.Aux[scala.Double, Double]

  type Array2D <: Any with Array2DApi.Aux[Companion, Array2D, Double, Boolean]
  implicit val Array2D: Companion[Array2D] with Lifter.Aux[Array[Array[scala.Double]], Array2D]

  def max(leftHandSide: Double, rightHandSide: Double): Double = {
    (leftHandSide < rightHandSide).`if`(rightHandSide)(leftHandSide)
  }

  def min(leftHandSide: Double, rightHandSide: Double): Double = {
    (leftHandSide < rightHandSide).`if`(leftHandSide)(rightHandSide)
  }

  def exp(array: Array2D): Array2D

  def log(array: Array2D): Array2D
}

object DeepLearning {

  object Differentiable {
    type Aux[+Data0, -Delta0] = Differentiable {
      type Data <: Data0
      type Delta >: Delta0
    }


    trait DifferentiableDouble extends Differentiable {

      override type Data = scala.Double

      override type Delta = scala.Double

      final def monoid: Monoid[Delta] = implicitly

    }


    trait DifferentiableBoolean extends Differentiable {

      override type Data = scala.Boolean

      override type Delta = scala.Boolean

      final def monoid = new Monoid[Delta] {
        override def empty: Boolean = false

        override def combine(x: Boolean, y: Boolean): Boolean = x ^ y
      }

    }

    trait DifferentiableArray2D extends Differentiable {

      override type Data = Eval[INDArray]

      override type Delta = Eval[Option[INDArray]]

      final def monoid = new Monoid[Delta] {
        override def empty: Eval[Option[INDArray]] = Eval.now(None)

        override def combine(x: Delta, y: Delta): Delta = Applicative[Eval].map2(x, y) {
          case (None, None) => None
          case (xDelta@Some(_), None) => xDelta
          case (None, yDelta@Some(_)) => yDelta
          case (Some(xDeltaValue), Some(yDeltaValue)) => Some(xDeltaValue add yDeltaValue)
        }
      }

    }


  }

  trait Differentiable {
    type Data
    type Delta

    def backward(delta: Delta): Unit

    def value: Data

  }

  object DifferentiableFunction {
    type Aux[-Input0 <: Differentiable, +Output0 <: Differentiable.Aux[_, _]] = DifferentiableFunction {
      type Input >: Input0
      type Output <: Output0
    }

    import Differentiable._

    final case class Id[Data0, Delta0]() extends DifferentiableFunction {
      outer =>
      type Input = Differentiable.Aux[Data0, Delta0]
      type Output = Differentiable.Aux[Data0, Delta0]

      override def forward(input: Input): Output = {
        input
      }
    }

    trait CachedFunction extends DifferentiableFunction {

      private val cache = java.util.Collections.synchronizedMap(new java.util.IdentityHashMap[Input, Output with ReferenceCount](1))

      trait ReferenceCount extends Differentiable {
        private[CachedFunction] var count: Int = 1

        implicit def monoid: Monoid[Delta]

        private var currentDelta: Delta = monoid.empty

        def input: Input

        protected def cachedBackward(delta: Delta): Unit

        override def backward(delta: Delta): Unit = {
          val (newDelta, newCount) = synchronized {
            count -= 1
            currentDelta = currentDelta |+| delta
            (currentDelta, count)
          }

          if (newCount == 0) {
            cache.remove(input)
            cachedBackward(newDelta)
          }
        }

      }

      type Output <: ReferenceCount

      protected def cachedForward(input: Input): Output

      override def forward(input: Input) = {
        cache.get(input) match {
          case null =>
            val output = cachedForward(input)
            cache.put(input, output)
            output
          case output =>
            output.synchronized {
              output.count += 1
            }
            output
        }
      }
    }

    import Differentiable._

    final case class Compose[A <: Differentiable, B <: Differentiable, C <: Differentiable](f: DifferentiableFunction.Aux[B, C], g: DifferentiableFunction.Aux[A, B]) extends DifferentiableFunction {
      override type Input = A
      override type Output = C

      override def forward(input: A): C = {
        f.forward(g.forward(input))
      }

    }

    final case class If[Input0 <: Differentiable, Output0 <: Differentiable](condition: DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Boolean, scala.Boolean]],
                                                                             `then`: DifferentiableFunction.Aux[Input0, Output0],
                                                                             `else`: DifferentiableFunction.Aux[Input0, Output0])
      extends DifferentiableFunction {
      override type Input = Input0
      override type Output = Output0

      override def forward(input: Input0): Output0 = {
        val conditionForwardPass = condition.forward(input)
        val output = if (conditionForwardPass.value) {
          `then`.forward(input)
        } else {
          `else`.forward(input)
        }
        conditionForwardPass.backward(false)
        output
      }
    }

    object DoubleFunction {
      type Aux[Input0] = DoubleFunction {
        type Input = Input0
      }

      final case class LessThan[Input0 <: Differentiable]
      (
        leftHandSide: DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Double, scala.Double]],
        rightHandSide: DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Double, scala.Double]]
      ) extends CachedFunction with BooleanFunction {

        final class Output(val input: Input0, upstream1: Differentiable.Aux[scala.Double, scala.Double], upstream2: Differentiable.Aux[scala.Double, scala.Double]) extends ReferenceCount with DifferentiableBoolean {
          type Input >: Input0
          val value = upstream1.value < upstream2.value

          override protected def cachedBackward(delta: scala.Boolean): Unit = {
            upstream1.backward(0.0)
            upstream2.backward(0.0)
          }
        }

        type Input = Input0

        override protected def cachedForward(input: Input): Output = {
          new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
        }
      }

      final case class Substract[Input0 <: Differentiable]
      (
        leftHandSide: DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Double, scala.Double]],
        rightHandSide: DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Double, scala.Double]]
      ) extends CachedFunction with DoubleFunction {

        final class Output(val input: Input0, upstream1: Differentiable.Aux[scala.Double, scala.Double], upstream2: Differentiable.Aux[scala.Double, scala.Double]) extends ReferenceCount with DifferentiableDouble {
          type Input >: Input0
          val value = upstream1.value - upstream2.value

          override protected def cachedBackward(delta: scala.Double): Unit = {
            upstream1.backward(delta)
            upstream2.backward(-delta)
          }
        }

        type Input = Input0

        override protected def cachedForward(input: Input): Output = {
          new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
        }

      }

      final case class Negative[Input0 <: Differentiable](toGeneric: DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Double, scala.Double]]) extends CachedFunction with DoubleFunction {

        final class Output(val input: Input0, upstream: Differentiable.Aux[scala.Double, scala.Double]) extends ReferenceCount with DifferentiableDouble {
          type Input >: Input0
          val value = -upstream.value


          override protected def cachedBackward(delta: scala.Double): Unit = {
            upstream.backward(-delta)
          }
        }

        type Input = Input0

        override protected def cachedForward(input: Input): Output = {
          val upstream = toGeneric.forward(input)
          new Output(input, upstream)
        }
      }

      final case class DoubleLiteral[Input0 <: Differentiable](value: scala.Double) extends DoubleFunction with DifferentiableDouble {
        override type Input = Input0
        override type Output = DoubleLiteral[Input0]

        override def forward(any: Input) = this

        override def backward(delta: Delta): Unit = {}

      }

      final case class DoubleWeight[Input0 <: Differentiable](var value: scala.Double)(implicit learningRate: LearningRate) extends DoubleFunction with DifferentiableDouble {
        override type Input = Input0
        override type Output = DoubleWeight[Input0]

        override def forward(any: Input) = this

        override def backward(delta: Delta): Unit = {
          value -= delta * learningRate()
        }

      }

      final case class DoubleOps[Input0 <: Differentiable](generic: DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Double, scala.Double]]) extends DoubleFunction {
        override type Output = generic.Output
        override type Input = Input0

        override def forward(input: Input0): generic.Output = {
          generic.forward(input)
        }
      }

    }

    trait DoubleFunction extends DifferentiableFunction with Dsl.DoubleApi {

      import DoubleFunction._

      override type Output <: Differentiable.Aux[scala.Double, scala.Double]

      override def unary_- : Double = new Negative(this)

      override def -(rightHandSide: Double): Double = new Substract(this, rightHandSide)

      override def <(rightHandSide: Double): Boolean = new LessThan(this, rightHandSide)

      override def +(rightHandSide: Double): Double = ???

      override def /(rightHandSide: Double): Double = ???

      override def /(rightHandSide: Array2D): Array2D = ???

      override def *(rightHandSide: Double): Double = ???
    }

    object Array2DFunction {
      type Aux[Input0] = Array2DFunction {
        type Input = Input0
      }

      final case class Array2DLiteral[Input0 <: Differentiable](rawValue: INDArray) extends Array2DFunction with DifferentiableArray2D {
        override type Input = Input0
        override type Output = Array2DLiteral[Input0]

        override def value = Eval.now(rawValue)

        override def forward(any: Input) = this

        override def backward(delta: Delta): Unit = {}
      }

      object Array2DLiteral {
        def apply[Input <: Differentiable](nativeArray: Array[Array[scala.Double]]): Array2DLiteral[Input] = {
          new Array2DLiteral[Input](nativeArray.toNDArray)
        }
      }

      final case class Array2DWeight[Input0 <: Differentiable](var rawValue: INDArray)(implicit learningRate: LearningRate) extends Array2DFunction with DifferentiableArray2D {
        override type Input = Input0
        override type Output = Array2DWeight[Input0]

        override def value = Eval.now(rawValue)

        override def forward(any: Input) = this

        override def backward(delta: Delta): Unit = {
          delta.value match {
            case Some(deltaValue) =>
              rawValue -= deltaValue * learningRate()
            case None =>
          }
        }
      }

      object Array2DWeight {
        def apply[Input <: Differentiable](nativeArray: Array[Array[scala.Double]])(implicit learningRate: LearningRate): Array2DWeight[Input] = {
          new Array2DWeight[Input](nativeArray.toNDArray)
        }
      }

      final case class Array2DOps[Input0 <: Differentiable](generic: DifferentiableFunction.Aux[Input0, Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Array2DFunction {
        override type Output = generic.Output
        override type Input = Input0

        override def forward(input: Input0): generic.Output = {
          generic.forward(input)
        }
      }

      final case class Dot[Input0 <: Differentiable]
      (
        leftHandSide: DifferentiableFunction.Aux[Input0, Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
        rightHandSide: DifferentiableFunction.Aux[Input0, Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]]]
      ) extends Array2DFunction with CachedFunction {

        final class Output(val input: Input0, upstream1: Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]], upstream2: Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with DifferentiableArray2D {
          type Input >: Input0
          val value = Applicative[Eval].map2(upstream1.value, upstream2.value)(_ dot _).memoize

          override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
            upstream1.backward(outputDelta.flatMap[Option[INDArray]] {
              case None => Eval.now(None)
              case Some(outputDeltaValue) =>
                upstream2.value.map {
                  bData =>
                    Some(outputDeltaValue.dot(bData.T))
                }
            }.memoize)
            upstream2.backward(outputDelta.flatMap[Option[INDArray]] {
              case None => Eval.now(None)
              case Some(outputDeltaValue) =>
                upstream1.value.map {
                  aData =>
                    Some(aData.T.dot(outputDeltaValue))
                }
            }.memoize)
          }
        }

        type Input = Input0

        override protected def cachedForward(input: Input): Output = {
          new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
        }
      }

      final case class MultiplyArray2D[Input0 <: Differentiable]
      (
        leftHandSide: DifferentiableFunction.Aux[Input0, Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
        rightHandSide: DifferentiableFunction.Aux[Input0, Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]]]
      ) extends Array2DFunction with CachedFunction {

        final class Output(val input: Input0, upstream1: Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]], upstream2: Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with DifferentiableArray2D {
          type Input >: Input0
          val value = Applicative[Eval].map2(upstream1.value, upstream2.value)(_ * _).memoize

          override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
            upstream1.backward(outputDelta.flatMap[Option[INDArray]] {
              case None => Eval.now(None)
              case Some(outputDeltaValue) =>
                upstream2.value.map {
                  bData =>
                    Some(bData * outputDeltaValue)
                }
            }.memoize)
            upstream2.backward(outputDelta.flatMap[Option[INDArray]] {
              case None => Eval.now(None)
              case Some(outputDeltaValue) =>
                upstream1.value.map {
                  aData =>
                    Some(aData * outputDeltaValue)
                }
            }.memoize)
          }
        }

        type Input = Input0

        override protected def cachedForward(input: Input): Output = {
          new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
        }
      }

      final case class AddArray2D[Input0 <: Differentiable]
      (
        leftHandSide: DifferentiableFunction.Aux[Input0, Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
        rightHandSide: DifferentiableFunction.Aux[Input0, Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]]]
      ) extends Array2DFunction with CachedFunction {

        final class Output(val input: Input0, upstream1: Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]], upstream2: Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with DifferentiableArray2D {
          type Input >: Input0
          val value = Applicative[Eval].map2(upstream1.value, upstream2.value)(_ + _).memoize

          override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
            upstream1.backward(outputDelta)
            upstream2.backward(outputDelta)
          }
        }

        type Input = Input0

        override protected def cachedForward(input: Input): Output = {
          new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
        }
      }


      final case class AddDouble[Input0 <: Differentiable]
      (
        leftHandSide: DifferentiableFunction.Aux[Input0, Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
        rightHandSide: DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Double, scala.Double]]
      ) extends Array2DFunction with CachedFunction {

        final class Output(val input: Input0, upstream1: Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]], upstream2: Differentiable.Aux[scala.Double, scala.Double]) extends ReferenceCount with DifferentiableArray2D {
          type Input >: Input0
          val value = upstream1.value.map(_ + upstream2.value).memoize

          override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
            upstream1.backward(outputDelta)
            upstream2.backward(outputDelta.value match {
              case None => 0.0
              case Some(deltaValue) => deltaValue.sumT
            })
          }
        }

        type Input = Input0

        override protected def cachedForward(input: Input): Output = {
          new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
        }
      }

      final case class Reciprocal[Input0 <: Differentiable](toGeneric: DifferentiableFunction.Aux[Input0, Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends CachedFunction with Array2DFunction {

        final class Output(val input: Input0, upstream: Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with DifferentiableArray2D {
          type Input >: Input0
          val value = upstream.value.map(_ rdiv 1.0).memoize


          override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
            upstream.backward(outputDelta.flatMap[Option[INDArray]] {
              case None => Eval.now(None)
              case Some(outputDeltaValue) =>
                upstream.value.map {
                  aValue: INDArray =>
                    Some(-outputDeltaValue / (aValue * aValue))
                }
            }.memoize)
          }
        }

        type Input = Input0

        override protected def cachedForward(input: Input): Output = {
          val upstream = toGeneric.forward(input)
          new Output(input, upstream)
        }
      }

      final case class Negative[Input0 <: Differentiable](toGeneric: DifferentiableFunction.Aux[Input0, Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends CachedFunction with Array2DFunction {

        final class Output(val input: Input0, upstream: Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with DifferentiableArray2D {
          type Input >: Input0
          val value = upstream.value.map(-_).memoize

          override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
            upstream.backward(outputDelta.map {
              case None => None
              case Some(outputDeltaValue) => Some(-outputDeltaValue)
            }.memoize)
          }
        }

        type Input = Input0

        override protected def cachedForward(input: Input): Output = {
          val upstream = toGeneric.forward(input)
          new Output(input, upstream)
        }
      }

    }

    trait Array2DFunction extends DifferentiableFunction with Dsl.Array2DApi {

      override type Output <: Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]]

      override def dot(rightHandSide: Array2D) = {
        Array2DFunction.Dot[Input](this, rightHandSide)
      }

      override def +(rightHandSide: Array2D) = {
        Array2DFunction.AddArray2D[Input](this, rightHandSide)
      }

      override def +(rightHandSide: Double): Array2D = {
        Array2DFunction.AddDouble[Input](this, rightHandSide)
      }

      override def /(rightHandSide: Array2D): Array2D = {
        Array2DFunction.MultiplyArray2D[Input](this, Array2DFunction.Reciprocal[Input](rightHandSide))
      }

      override def /(rightHandSide: Double): Array2D = ???

      override def *(rightHandSide: Array2D) = {
        Array2DFunction.MultiplyArray2D[Input](this, rightHandSide)
      }

      override def *(rightHandSide: Double): Array2D = ???

      override def unary_- = {
        Array2DFunction.Negative[Input](this)
      }
    }

    object BooleanFunction {
      type Aux[Input0] = BooleanFunction {
        type Input = Input0
      }

      final case class BooleanOps[Input0 <: Differentiable](generic: DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Boolean, scala.Boolean]]) extends BooleanFunction {
        override type Output = generic.Output
        override type Input = Input0

        override def forward(input: Input0): generic.Output = {
          generic.forward(input)
        }
      }

    }

    trait BooleanFunction extends DifferentiableFunction with Dsl.BooleanApi {
      type Output <: Differentiable.Aux[scala.Boolean, scala.Boolean]

      override def `if`[A](`then`: A)(`else`: A)(implicit companion: Companion[A]): A = {
        companion.specialize(If[Input, companion.Output](this, companion.generalize(`then`), companion.generalize(`else`)))
      }

      override def unary_! : Boolean = ???
    }

    object DifferentiableFunctionCompanion {
      type Aux[Input0, Output0, SpecialFunction0] = DifferentiableFunctionCompanion {
        type Input = Input0
        type Output = Output0
        type SpecialFunction = SpecialFunction0
      }
    }

    trait DifferentiableFunctionCompanion {
      type Input <: Differentiable
      type Output <: Differentiable
      type SpecialFunction

      /**
        * Returns the base [[DeepLearning.DifferentiableFunction]] type of a [[SpecialFunction]].
        */
      def generalize(specialFunction: SpecialFunction): DifferentiableFunction.Aux[Input, Output]

      /**
        * Returns a special subclass of a [[DeepLearning.DifferentiableFunction]].
        */
      def specialize(generic: DifferentiableFunction.Aux[Input, Output]): SpecialFunction
    }

  }

  trait DifferentiableFunction {
    outer =>

    import DifferentiableFunction._

    type Companion[SpecialFunction0] = DifferentiableFunctionCompanion {
      type SpecialFunction = SpecialFunction0
      type Input = DifferentiableFunction.this.Input
    }
    type Array2D = Array2DFunction.Aux[Input]
    type Double = DoubleFunction.Aux[Input]
    type Boolean = BooleanFunction.Aux[Input]

    type Input <: Differentiable

    type Output <: Differentiable.Aux[_, _]

    def forward(input: Input): Output

  }

  trait LearningRate {
    def apply(): scala.Double
  }

  import DifferentiableFunction._

}

import DeepLearning._

final class DeepLearning[Input0 <: Differentiable](implicit learningRate: LearningRate) extends Dsl {

  import DifferentiableFunction._

  override type Companion[SpecialFunction0] = DifferentiableFunctionCompanion {
    type SpecialFunction = SpecialFunction0
    type Input = Input0
  }

  override type Any = DifferentiableFunction.Aux[Input0, Differentiable.Aux[_, _]]

  override object Any extends DifferentiableFunctionCompanion {
    override type SpecialFunction = Any
    override type Input = Input0
    override type Output = Differentiable.Aux[_, _]

    override def generalize(generic: Any): Any = generic

    override def specialize(generic: Any): Any = generic
  }

  override type Double = DoubleFunction.Aux[Input0]

  implicit override object Double extends Dsl.Lifter with DifferentiableFunctionCompanion {

    import DoubleFunction._

    override type LiftFrom = scala.Double
    override type LiftTo = Double
    override type SpecialFunction = Double
    override type Input = Input0
    override type Output = Differentiable.Aux[scala.Double, scala.Double]

    override def apply(value: scala.Double) = DoubleLiteral[Input0](value)

    override def weight(initialValue: scala.Double) = DoubleWeight[Input0](initialValue)

    override def specialize(generic: DifferentiableFunction.Aux[Input0, Output]) = DoubleOps(generic)

    override def generalize(generic: Double): Double = generic

  }

  override type Array2D = Array2DFunction.Aux[Input0]

  implicit override object Array2D extends Dsl.Lifter with DifferentiableFunctionCompanion {

    import Array2DFunction._

    override type LiftFrom = Array[Array[scala.Double]]
    override type LiftTo = Array2D
    override type SpecialFunction = Array2D
    override type Input = Input0
    override type Output = Differentiable.Aux[Eval[INDArray], Eval[Option[INDArray]]]

    override def apply(value: Array[Array[scala.Double]]) = Array2DFunction.Array2DLiteral[Input0](value)

    override def weight(initialValue: Array[Array[scala.Double]]) = Array2DWeight[Input0](initialValue)

    override def generalize(specialFunction: Array2D) = specialFunction

    override def specialize(generic: DifferentiableFunction.Aux[Input0, Output]) = Array2DOps(generic)

  }

  override type Boolean = BooleanFunction.Aux[Input0]

  implicit override object Boolean extends DifferentiableFunctionCompanion {

    import BooleanFunction._

    override type SpecialFunction = Boolean
    override type Input = Input0
    override type Output = Differentiable.Aux[scala.Boolean, scala.Boolean]

    override def generalize(generic: Boolean): Boolean = generic

    override def specialize(generic: DifferentiableFunction.Aux[Input0, Output]) = BooleanOps(generic)

  }

  override def exp(array: Array2D): Array2D = ???

  override def log(array: Array2D): Array2D = ???
}
