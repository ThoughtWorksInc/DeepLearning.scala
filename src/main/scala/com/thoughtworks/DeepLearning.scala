package com.thoughtworks

import cats.Monoid

import scala.language.existentials
import scala.language.implicitConversions
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

  trait BooleanOps[Any <: {type Self <: Any}, Boolean <: Any] {
    def `if`[A <: Any](`then`: A)(`else`: A): A#Self
  }

  trait DoubleExtractor[Double] extends (scala.Double => Double) {
    def weight(value: scala.Double): Double
  }

}

trait Dsl {

  import Dsl._

  type Any <: {
    type Self <: Any
  }

  type Boolean <: Any {
    type Self <: Boolean
  }

  implicit def booleanOps(underlying: Boolean): BooleanOps[Any, Boolean]

  type Double <: Any {
    type Self <: Double
  }

  val Double: DoubleExtractor[Double]

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

    final class Id[Data0, Delta0] extends DifferentiableFunction {
      outer =>
      type Input = Differentiable.Aux[Data0, Delta0]
      type Output = Differentiable.Aux[Data0, Delta0]

      override def forward(input: Input): Output = {
        input
      }
    }


    trait CachedFunction extends DifferentiableFunction {

      private val cache = new java.util.concurrent.ConcurrentHashMap[Input, Output with ReferenceCount](1)

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

    final class NegativeFunction[Input0 <: Differentiable](from: DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Double, scala.Double]]) extends CachedFunction {

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

    final class SubstractFunction[Input0 <: Differentiable]
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

  }

  trait DifferentiableFunction {
    outer =>

    type Input <: Differentiable

    type Output <: Differentiable.Aux[_, _]

    type Self >: this.type <: DifferentiableFunction.Aux[Input, Output]

    def self: Self = this

    def forward(input: Input): Output

    final def compose[Input0 <: Differentiable](f: DifferentiableFunction.Aux[Input0, Input]) = {
      new DifferentiableFunction {
        override type Input = Input0
        override type Output = outer.Output

        override def forward(input: Input0): Output = {
          outer.forward(f.forward(input))
        }
      }
    }

  }

  //  object If {
  //
  //    sealed trait IfState[Data, Delta] {
  //      val data: Data
  //      val delta: Delta
  //      val count: Int
  //    }
  //
  //    final case class Then[Data, Delta](data: Data, delta: Delta, count: Int) extends IfState[Data, Delta]
  //
  //    final case class Else[Data, Delta](data: Data, delta: Delta, count: Int) extends IfState[Data, Delta]
  //
  //  }
  //
  //  import If._
  //
  //  final case class If[Input0 <: DifferentiableFunction, Data0, Delta0](condition: DifferentiableFunction.Aux[_ >: Input0, scala.Boolean, scala.Boolean],
  //                                                               `then`: DifferentiableFunction.Aux[_ >: Input0, Data0, Delta0],
  //                                                               `else`: DifferentiableFunction.Aux[_ >: Input0, Data0, Delta0])
  //    extends DifferentiableFunction {
  //
  //    override type Data = Data0
  //    override type Delta = Delta0
  //    override type Input = Input0
  //
  //    private val cache = collection.mutable.HashMap.empty[Input, IfState[Data, Delta]]
  //
  //    implicit override def monoid = `then`.monoid
  //
  //    override def predict(input: Input): Data0 = {
  //      synchronized {
  //        cache.get(input)
  //      } match {
  //        case None =>
  //          if (condition.predict(input)) {
  //            val newData = `then`.predict(input)
  //            synchronized {
  //              cache.put(input, Then(newData, monoid.empty, 1))
  //            }
  //            newData
  //          } else {
  //            val newData = `else`.predict(input)
  //            synchronized {
  //              cache.put(input, Else(newData, monoid.empty, 1))
  //            }
  //            newData
  //          }
  //        case Some(Else(data, delta, count)) =>
  //          synchronized {
  //            cache.put(input, Else(data, delta, count + 1))
  //          }
  //          data
  //        case Some(Then(data, delta, count)) =>
  //          synchronized {
  //            cache.put(input, Then(data, delta, count + 1))
  //          }
  //          data
  //      }
  //    }
  //
  //    override def train(input: Input, delta: Delta0): Unit = {
  //
  //      synchronized {
  //        cache(input)
  //      } match {
  //        case Then(data, originalDelta, count) =>
  //          val newDelta = delta |+| originalDelta
  //          if (count == 1) {
  //            synchronized {
  //              cache.remove(input)
  //            }
  //            `then`.train(input, newDelta)
  //            condition.train(input, condition.monoid.empty)
  //          } else {
  //            synchronized {
  //              cache.put(input, Then(data, newDelta, count - 1))
  //            }
  //          }
  //        case Else(data, originalDelta, count) =>
  //          val newDelta = delta |+| originalDelta
  //          if (count == 1) {
  //            synchronized {
  //              cache.remove(input)
  //            }
  //            `else`.train(input, newDelta)
  //            condition.train(input, condition.monoid.empty)
  //          } else {
  //            synchronized {
  //              cache.put(input, Else(data, newDelta, count - 1))
  //            }
  //          }
  //      }
  //    }
  //  }
  //
  trait LearningRate {
    def apply(): scala.Double
  }

}


import DeepLearning._

final class DeepLearning[Input0 <: Differentiable](implicit learningRate: LearningRate) extends Dsl {

  import DifferentiableFunction._
  import Differentiable._

  override type Any = DifferentiableFunction.Aux[Input0, Differentiable.Aux[_, _]]

  override type Double = DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Double, scala.Double]]

  override object Double extends Dsl.DoubleExtractor[Double] {
    override def apply(initialData: scala.Double) = Literal(initialData)

    override def weight(initialData: scala.Double) = DoubleWeight(initialData)
  }

  override type Boolean = DifferentiableFunction.Aux[Input0, Differentiable.Aux[scala.Boolean, scala.Boolean]]

  override def doubleOps(underlying: Double) = new Dsl.DoubleOps[Double] {
    override def unary_- = new NegativeFunction(underlying)

    override def -(rightHandSide: Double) = new SubstractFunction(underlying, rightHandSide)

  }

  override def booleanOps(underlying: Boolean) = ???

  //
  //  def booleanOps(underlying: Boolean) = new BooleanOps[Any, Boolean] {
  //    override def `if`[A <: Any](`then`: A)(`else`: A) = {
  //      If[Input0, A#Data, A#Delta](underlying, `then`, `else`).self
  //    }
  //  }

}
