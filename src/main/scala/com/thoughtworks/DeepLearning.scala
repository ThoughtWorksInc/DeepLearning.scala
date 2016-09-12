package com.thoughtworks

import cats.Monoid

import scala.language.existentials
import scala.language.implicitConversions
import scala.language.higherKinds
import cats.implicits._

object Dsl {

  trait DoubleOps[Double] {
    def unary_- : Double

    def -(rightHandSide: Double): Double
  }

  object Double {
    type Aux[Self0] = Double {
      type Self = Self0
    }
  }


  object BooleanApi {
    type Aux[Any0, Companion0[_]] = BooleanApi {
      type Any = Any0
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


  type Any
  type Companion[_]
  implicit val Any: Companion[Any]

  type Boolean <: Any with BooleanApi {
    type Companion[A] = Dsl.this.Companion[A]
  }

  implicit val Boolean: Companion[Boolean]

  type Double <: Any {
    type Self <: Double
  }

  implicit val Double: Companion[Double] with DoubleExtractor[Double]

  implicit def doubleOps(underlying: Double): DoubleOps[Double]

}

object DeepLearning {

  object Differentiable {
    type Aux[+Data0, -Delta0] = Differentiable {
      type Data <: Data0
      type Delta >: Delta0
    }


    trait DifferentiableDouble extends Differentiable {

      override type Self >: this.type <: DifferentiableDouble

      override type Data = scala.Double

      override type Delta = scala.Double

      final def monoid: Monoid[Delta] = implicitly

    }

  }

  trait Differentiable {
    type Data
    type Delta
    type Self >: this.type <: Differentiable.Aux[Data, Delta]

    def backward(delta: Delta): Unit

    def value: Data

    def self: Self = this
  }

  object DifferentiableFunction {
    type Aux[-Input0 <: Differentiable, +Output0 <: Differentiable.Aux[_, _]] = DifferentiableFunction {
      type Input >: Input0
      type Output <: Output0
    }

    final case class Literal[Data0](value: Data0) extends DifferentiableFunction with Differentiable {
      override type Data = Data0
      override type Delta = Any
      override type Input = Differentiable
      override type Output = Literal[Data0]
      override type Self = Literal[Data0]

      override def self: Self = this

      override def forward(any: Input) = this

      override def backward(delta: Delta): Unit = {}

    }

    final case class DoubleWeight(var value: scala.Double)(implicit learningRate: LearningRate) extends DifferentiableFunction with Differentiable {
      override type Data = scala.Double
      override type Delta = scala.Double
      override type Input = Differentiable
      override type Output = DoubleWeight
      override type Self = DoubleWeight

      override def self: Self = this

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

    final case class Negative[Input0 <: Differentiable](from: DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Double, scala.Double]]) extends CachedFunction {

      final class Output(val input: Input0, upstream: Differentiable.Aux[scala.Double, scala.Double]) extends ReferenceCount with DifferentiableDouble {
        type Input >: Input0
        val value = -upstream.value


        override protected def cachedBackward(input: Input0, delta: scala.Double): Unit = {
          upstream.backward(-delta)
        }
      }

      type Input = Input0

      override protected def cachedForward(input: Input): Output = {
        val upstream = from.forward(input)
        new Output(input, upstream)
      }
    }

    final case class Substract[Input0 <: Differentiable]
    (
      leftHandSide: DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Double, scala.Double]],
      rightHandSide: DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Double, scala.Double]]
    ) extends CachedFunction {

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
      override type Self = If[Input0, Output0]
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

    object BooleanFunction {
      type Aux[Input0] = BooleanFunction {
        type Input = Input0
      }
    }

    trait BooleanFunction extends DifferentiableFunction with Dsl.BooleanApi {
      type Output <: Differentiable.Aux[scala.Boolean, scala.Boolean]

      override def `if`[A](`then`: A)(`else`: A)(implicit companion: Companion[A]): A = {
        companion.to(If[Input, companion.Output](this, companion.from(`then`), companion.from(`else`)))
      }
    }

    trait DifferentiableFunctionCompanion {
      type Input <: Differentiable
      type Output <: Differentiable
      type Target

      def from(target: Target): DifferentiableFunction.Aux[Input, Output]

      def to(generic: DifferentiableFunction.Aux[Input, Output]): Target

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

    type Self >: this.type <: DifferentiableFunction.Aux[Input, Output]

    def self: Self = this

    def forward(input: Input): Output

  }

  trait LearningRate {
    def apply(): scala.Double
  }


}


import DeepLearning._

final class DeepLearning[Input0 <: Differentiable](implicit learningRate: LearningRate) extends Dsl {

  import DifferentiableFunction._
  import Differentiable._

  override type Any = DifferentiableFunction.Aux[Input0, Differentiable.Aux[_, _]]

  type Companion[Target0] = DifferentiableFunctionCompanion {
    type Target = Target0
    type Input = Input0
  }

  override type Double = DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Double, scala.Double]]

  override object Double extends Dsl.DoubleExtractor[Double] with DifferentiableFunctionCompanion {
    override type Target = Double
    override type Input = Input0
    override type Output = Differentiable.Aux[scala.Double, scala.Double]

    override def apply(value: scala.Double) = Literal(value)

    override def weight(initialValue: scala.Double) = DoubleWeight(initialValue)

    override def to(generic: Double): Double = generic

    override def from(generic: Double): Double = generic

  }

  override type Boolean = BooleanFunction.Aux[Input0]

  override def doubleOps(underlying: Double) = new Dsl.DoubleOps[Double] {
    override def unary_- = new Negative(underlying)

    override def -(rightHandSide: Double) = new Substract(underlying, rightHandSide)

  }

  object Any extends DifferentiableFunctionCompanion {
    override type Target = Any
    override type Input = Input0
    override type Output = Differentiable.Aux[_, _]

    override def from(generic: Any): Any = generic

    override def to(generic: Any): Any = generic
  }


  object Boolean extends DifferentiableFunctionCompanion {
    override type Target = Boolean
    override type Input = Input0
    override type Output = Differentiable.Aux[scala.Boolean, scala.Boolean]

    override def from(generic: Boolean): Boolean = generic

    override def to(generic: DifferentiableFunction.Aux[Input0, Output]) = {
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
