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

  object Cache {
    type Aux[+Data0, -Delta0] = Cache {
      type Data <: Data0
      type Delta >: Delta0
    }
  }

  trait Cache {
    type Data
    type Delta
    type Self >: this.type <: Cache.Aux[Data, Delta]

    def backward(delta: Delta): Unit

    def value: Data

    def self: Self = this
  }


  trait DoubleCache extends Cache {

    override type Self >: this.type <: DoubleCache

    override type Data = scala.Double

    override type Delta = scala.Double

    final def monoid: Monoid[Delta] = implicitly

  }

  object Differentiable {
    type Aux[-Input0 <: Cache, +Output0 <: Cache.Aux[_, _]] = Differentiable {
      type Input >: Input0
      type Output <: Output0
    }

    def compose[Input0 <: Cache, Input1 >: Cache.Aux[_, _] <: Cache, OutputData1, OutputDelta1]
    (
      f: Differentiable.Aux[Input1, Cache.Aux[OutputData1, OutputDelta1]],
      g: Differentiable.Aux[Input0, Cache.Aux[_, _]]
    ) = {
      new Differentiable {
        override type Output = Cache.Aux[OutputData1, OutputDelta1]
        override type Input = Input0

        override def forward(input: Input): Output = {
          val c = f.forward(g.forward(input))

          abstract class CacheImpl[C <: Cache](val c: C) extends Cache {
            override type Data = c.Data
            override type Delta = c.Delta
            override type Self = Cache.Aux[c.Data, c.Delta]

            override def value = c.value
          }

          new CacheImpl[c.type](c) {
            override def backward(delta: Delta): Unit = {
              c.backward(delta)
            }
          }
        }
      }
    }

  }

  trait Differentiable {
    outer =>

    type Input <: Cache

    type Output <: Cache.Aux[_, _]

    type Self >: this.type <: Differentiable.Aux[Input, Output]

    def self: Self = this

    def forward(input: Input): Output

    final def compose[Input0 <: Cache](f: Differentiable.Aux[Input0, Input]) = {
      new Differentiable {
        override type Input = Input0
        override type Output = outer.Output

        override def forward(input: Input0): Output = {
          outer.forward(f.forward(input))
        }
      }
    }

  }

  final class Id[Data0, Delta0] extends Differentiable {
    outer =>
    type Input = Cache.Aux[Data0, Delta0]
    type Output = Cache.Aux[Data0, Delta0]

    override def forward(input: Input): Output = {
      input
    }
  }

  //
  //  def doubleId = new Id[DoubleCache.Aux[DoubleCache]]
  //
  //  def xxx = {
  //    doubleId.forward(new DoubleCache {
  //      override def value = 1.0
  //      override def backward(input: Input, delta: Double): Unit = {}
  //
  //      override type Input = Cache
  //    })
  //  }


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
  //  final case class If[Input0 <: Differentiable, Data0, Delta0](condition: Differentiable.Aux[_ >: Input0, scala.Boolean, scala.Boolean],
  //                                                               `then`: Differentiable.Aux[_ >: Input0, Data0, Delta0],
  //                                                               `else`: Differentiable.Aux[_ >: Input0, Data0, Delta0])
  //    extends Differentiable {
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
  //  final case class Constant[@specialized Data0, Delta0](data: Data0)(implicit val monoid: Monoid[Delta0]) extends Differentiable {
  //    override type Input = Differentiable
  //    override type Data = Data0
  //    override type Delta = Delta0
  //
  //    def train(input: Input, delta: Delta): Unit = {}
  //
  //    def predict(input: Input) = data
  //  }
  //
  //
  //     abstract class Weight[@specialized Data0, Delta0](@volatile protected var data: Data0)(implicit val monoid: Monoid[Delta0]) extends Differentiable {
  //
  //       override type Input = Differentiable
  //
  // //      override type Data = Data0
  // //      override type Delta = Delta0
  //
  // //      final def predict(input: Input) = data
  //
  //     }

  trait Cached extends Differentiable {

    private val cache = new java.util.concurrent.ConcurrentHashMap[Input, Output with ReferenceCount](1)

    trait ReferenceCount extends Cache {
      private[Cached] var count: Int = 1

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
        case output =>
          output.synchronized {
            output.count += 1
          }
          output
      }
    }
  }

  trait LearningRate {
    def apply(): scala.Double
  }

  final case class Literal[Data0](value: Data0) extends Differentiable with Cache {
    override type Data = Data0
    override type Delta = Any
    override type Input = Cache
    override type Output = Literal[Data0]
    override type Self = Literal[Data0]

    override def self: Self = this

    override def forward(any: Input) = this

    override def backward(delta: Delta): Unit = {}

  }

  final case class DoubleWeight(var value: scala.Double)(implicit learningRate: LearningRate) extends Differentiable with Cache {
    override type Data = scala.Double
    override type Delta = scala.Double
    override type Input = Cache
    override type Output = DoubleWeight
    override type Self = DoubleWeight

    override def self: Self = this

    override def forward(any: Input) = this

    override def backward(delta: Delta): Unit = {
      value -= delta * learningRate()
    }

  }

}


import DeepLearning._

final class DeepLearning[Input0 <: Cache](implicit learningRate: LearningRate) extends Dsl {

  override type Any = Differentiable.Aux[Input0, Cache.Aux[_, _]]

  override type Double = Differentiable.Aux[Input0, Cache.Aux[scala.Double, scala.Double]]

  override object Double extends Dsl.DoubleExtractor[Double] {
    override def apply(initialData: scala.Double) = Literal(initialData)

    override def weight(initialData: scala.Double) = DoubleWeight(initialData)
  }

  override type Boolean = Differentiable.Aux[Input0, Cache.Aux[scala.Boolean, scala.Boolean]]

  override def doubleOps(underlying: Double) = new Dsl.DoubleOps[Double] {
    override def unary_- = {
      new Cached {

        final class Output(val input: Input0, upstream: Cache.Aux[scala.Double, scala.Double]) extends ReferenceCount with DoubleCache {
          type Input >: Input0
          val value = -upstream.value


          override protected def cachedBackward(input: Input0, delta: scala.Double): Unit = {
            upstream.backward(-delta)
          }
        }

        type Input = Input0

        override protected def cachedForward(input: Input): Output = {
          val upstream = underlying.forward(input)
          new Output(input, upstream)
        }
      }
    }

    override def -(rightHandSide: Double) = {
      new Cached {

        final class Output(val input: Input0, upstream1: Cache.Aux[scala.Double, scala.Double], upstream2: Cache.Aux[scala.Double, scala.Double]) extends ReferenceCount with DoubleCache {
          type Input >: Input0
          val value = upstream1.value - upstream2.value

          override protected def cachedBackward(input: Input0, delta: scala.Double): Unit = {
            upstream1.backward(delta)
            upstream2.backward(-delta)
          }
        }

        type Input = Input0

        override protected def cachedForward(input: Input): Output = {
          val upstream = underlying.forward(input)
          new Output(input, underlying.forward(input), rightHandSide.forward(input))
        }

      }

    }

  }

  override def booleanOps(underlying: Boolean) = ???

  //
  //  def booleanOps(underlying: Boolean) = new BooleanOps[Any, Boolean] {
  //    override def `if`[A <: Any](`then`: A)(`else`: A) = {
  //      If[Input0, A#Data, A#Delta](underlying, `then`, `else`).self
  //    }
  //  }

}
