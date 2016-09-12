package com.thoughtworks

import cats.Monoid

import scala.language.existentials
import scala.language.implicitConversions
import scala.language.higherKinds
import cats.implicits._

object Dsl {

  trait DoubleApi {
    type Double <: DoubleApi
    type Boolean <: BooleanApi

    def unary_- : Double

    def -(rightHandSide: Double): Double

    def <(rightHandSide: Double): Boolean

  }

  object DoubleApi {
    type Aux[Double0, Boolean0] = DoubleApi {
      type Double = Double0
      type Boolean = Boolean0
    }
  }


  object BooleanApi {
    type Aux[Companion0[_]] = BooleanApi {
      type Companion[A] = Companion0[A]
    }
  }


  trait BooleanApi {
    type Companion[_]

    def `if`[A: Companion](`then`: A)(`else`: A): A
  }

  trait DoubleExtractor[Double] extends (scala.Double => Double) {
    def weight(initialValue: scala.Double): Double
  }

}

trait Dsl {

  import Dsl._

  type Companion[_]

  type Any
  implicit val Any: Companion[Any]

  type Boolean <: Any with BooleanApi.Aux[Companion]
  implicit val Boolean: Companion[Boolean]

  type Double <: Any with DoubleApi.Aux[Double, Boolean]

  implicit val Double: Companion[Double] with DoubleExtractor[Double]

  def max(leftHandSide: Double, rightHandSide: Double): Double = {
    (leftHandSide < rightHandSide).`if`(rightHandSide)(leftHandSide)
  }

  def min(leftHandSide: Double, rightHandSide: Double): Double = {
    (leftHandSide < rightHandSide).`if`(leftHandSide)(rightHandSide)
  }
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

        protected def cachedBackward(input: Input, delta: Delta): Unit

        override def backward(delta: Delta): Unit = {
          val (newDelta, newCount) = synchronized {
            count -= 1
            currentDelta = currentDelta |+| delta
            (currentDelta, count)
          }

          if (newCount == 0) {
            cache.remove(input)
            cachedBackward(input, newDelta)
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

    final case class Negative[Input0 <: Differentiable](toGeneric: DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Double, scala.Double]]) extends CachedFunction with DoubleFunction {

      final class Output(val input: Input0, upstream: Differentiable.Aux[scala.Double, scala.Double]) extends ReferenceCount with DifferentiableDouble {
        type Input >: Input0
        val value = -upstream.value


        override protected def cachedBackward(input: Input0, delta: scala.Double): Unit = {
          upstream.backward(-delta)
        }
      }

      type Input = Input0

      override protected def cachedForward(input: Input): Output = {
        val upstream = toGeneric.forward(input)
        new Output(input, upstream)
      }
    }

    final case class LessThan[Input0 <: Differentiable]
    (
      leftHandSide: DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Double, scala.Double]],
      rightHandSide: DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Double, scala.Double]]
    ) extends CachedFunction with BooleanFunction {

      final class Output(val input: Input0, upstream1: Differentiable.Aux[scala.Double, scala.Double], upstream2: Differentiable.Aux[scala.Double, scala.Double]) extends ReferenceCount with DifferentiableBoolean {
        type Input >: Input0
        val value = upstream1.value < upstream2.value

        override protected def cachedBackward(input: Input0, delta: scala.Boolean): Unit = {
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

        override protected def cachedBackward(input: Input0, delta: scala.Double): Unit = {
          upstream1.backward(delta)
          upstream2.backward(-delta)
        }
      }

      type Input = Input0

      override protected def cachedForward(input: Input): Output = {
        new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
      }

    }

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
    }

    final case class DoubleOps[Input0 <: Differentiable](generic: DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Double, scala.Double]]) extends DoubleFunction {
      type Output = generic.Output
      type Input = Input0

      override def forward(input: Input0): generic.Output = {
        generic.forward(input)
      }
    }


    trait DoubleFunction extends DifferentiableFunction with Dsl.DoubleApi {
      override type Double = DoubleFunction.Aux[Input]
      override type Boolean = BooleanFunction.Aux[Input]
      override type Output <: Differentiable.Aux[scala.Double, scala.Double]

      override def unary_- : Double = new Negative(this)

      override def -(rightHandSide: Double): Double = new Substract(this, rightHandSide)

      override def <(rightHandSide: Double): Boolean = new LessThan(this, rightHandSide)
    }

    object BooleanFunction {
      type Aux[Input0] = BooleanFunction {
        type Input = Input0
      }
    }

    trait BooleanFunction extends DifferentiableFunction with Dsl.BooleanApi {
      type Output <: Differentiable.Aux[scala.Boolean, scala.Boolean]

      override def `if`[A](`then`: A)(`else`: A)(implicit companion: Companion[A]): A = {
        companion.fromGeneric(If[Input, companion.Output](this, companion.toGeneric(`then`), companion.toGeneric(`else`)))
      }
    }

    object DifferentiableFunctionCompanion {
      type Aux[Input0, Output0, Target0] = DifferentiableFunctionCompanion {
        type Input = Input0
        type Output = Output0
        type Target = Target0
      }
    }

    trait DifferentiableFunctionCompanion {
      type Input <: Differentiable
      type Output <: Differentiable
      type Target

      def toGeneric(target: Target): DifferentiableFunction.Aux[Input, Output]

      def fromGeneric(generic: DifferentiableFunction.Aux[Input, Output]): Target
    }

  }

  trait DifferentiableFunction {
    outer =>

    import DifferentiableFunction._

    type Companion[Target0] = DifferentiableFunctionCompanion {
      type Target = Target0
      type Input = DifferentiableFunction.this.Input
    }

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

  override type Any = DifferentiableFunction.Aux[Input0, Differentiable.Aux[_, _]]

  type Companion[Target0] = DifferentiableFunctionCompanion {
    type Target = Target0
    type Input = Input0
  }

  override type Double = DoubleFunction.Aux[Input0]

  override object Double extends Dsl.DoubleExtractor[Double] with DifferentiableFunctionCompanion {
    override type Target = Double
    override type Input = Input0
    override type Output = Differentiable.Aux[scala.Double, scala.Double]

    override def apply(value: scala.Double) = DoubleLiteral[Input0](value)

    override def weight(initialValue: scala.Double) = DoubleWeight[Input0](initialValue)

    override def fromGeneric(generic: DifferentiableFunction.Aux[Input0, Output]) = {
      DoubleOps(generic)
    }


    override def toGeneric(generic: Double): Double = generic

  }

  override type Boolean = BooleanFunction.Aux[Input0]

  object Any extends DifferentiableFunctionCompanion {
    override type Target = Any
    override type Input = Input0
    override type Output = Differentiable.Aux[_, _]

    override def toGeneric(generic: Any): Any = generic

    override def fromGeneric(generic: Any): Any = generic
  }


  object Boolean extends DifferentiableFunctionCompanion {
    override type Target = Boolean
    override type Input = Input0
    override type Output = Differentiable.Aux[scala.Boolean, scala.Boolean]

    override def toGeneric(generic: Boolean): Boolean = generic

    override def fromGeneric(generic: DifferentiableFunction.Aux[Input0, Output]) = {
      new BooleanFunction {
        type Output = generic.Output
        type Input = Input0

        override def forward(input: Input0): generic.Output = {
          generic.forward(input)
        }
      }
    }

  }

}
