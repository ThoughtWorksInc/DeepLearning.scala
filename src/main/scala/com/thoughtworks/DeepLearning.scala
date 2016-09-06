package com.thoughtworks


import cats.Monoid
import com.thoughtworks.DeepLearning.Differentiable._

import scala.language.existentials
import com.thoughtworks.Dsl.{BooleanOps, DoubleOps}

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

}

trait Dsl {

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

  val Double: scala.Double => Double

  implicit def doubleOps(underlying: Double): DoubleOps[Double]
}


object DeepLearning extends Dsl {

  object Differentiable {

    trait MiniBatch

    final case class CacheState[Data, Delta](data: Data, delta: Delta, count: Int)

    trait Cached extends Differentiable {

      def cache = new collection.mutable.HashMap[MiniBatch, CacheState[Data, Delta]]

      def forward(miniBatch: MiniBatch): Data

      final def predict(miniBatch: MiniBatch): Data = {
        synchronized {
          cache.get(miniBatch)
        } match {
          case None =>
            val newData = forward(miniBatch)
            synchronized {
              cache.put(miniBatch, CacheState(newData, monoid.empty, 1))
            }
            newData
          case Some(CacheState(data, delta, count)) =>
            synchronized {
              cache.put(miniBatch, CacheState(data, delta, count + 1))
            }
            data
        }
      }

      def backward(miniBatch: MiniBatch, data: Data, delta: Delta): Unit

      final def train(miniBatch: MiniBatch, delta: Delta): Unit = {
        synchronized {
          cache(miniBatch)
        } match {
          case CacheState(data, originalDelta, count) =>
            val newDelta = delta |+| originalDelta
            if (count == 1) {
              synchronized {
                cache.remove(miniBatch)
              }
              backward(miniBatch, data, newDelta)
            } else {
              synchronized {
                cache.put(miniBatch, CacheState(data, newDelta, count - 1))
              }
            }
        }
      }
    }

    type Aux[Data0, Delta0] = Differentiable {
      type Data = Data0
      type Delta = Delta0
    }

  }

  trait Differentiable {
    outer =>

    type Self = Differentiable.Aux[Data, Delta]

    type Data
    type Delta

    import Differentiable._

    def predict(miniBatch: MiniBatch): Data

    def train(miniBatch: MiniBatch, delta: Delta): Unit

    final def self: Self = this

    implicit def monoid: Monoid[Delta]

  }

  trait DifferentiableDouble extends Differentiable {

    override type Data = scala.Double
    override type Delta = scala.Double

    override final def monoid = cats.instances.double.catsKernelStdGroupForDouble

  }

  trait DifferentiableBoolean extends Differentiable {

    override type Data = scala.Boolean

    override type Delta = scala.Boolean

    override final def monoid = new Monoid[scala.Boolean] {
      override def empty = false

      override def combine(delta0: scala.Boolean, delta1: scala.Boolean) = delta0 ^ delta1
    }
  }

  override type Any = Differentiable.Aux[_, _]

  override type Double = Differentiable.Aux[scala.Double, scala.Double]

  override object Double extends (scala.Double => Double) {
    def apply(initialData: scala.Double) = new DifferentiableDouble {
      @volatile var data = initialData
      def predict(miniBatch: MiniBatch) = data
      def train(miniBatch: MiniBatch, delta: scala.Double): Unit = {
        synchronized {
          data += delta
        }
      }
    }
  }

  override type Boolean = Differentiable.Aux[scala.Boolean, scala.Boolean]

  final case class Input[Data0, Delta0](implicit val monoid: Monoid[Delta0]) extends Differentiable {
    type Data = Data0
    type Delta = Delta0

    val data = collection.mutable.HashMap.empty[MiniBatch, Data]

    override def predict(miniBatch: MiniBatch): Data = {
      data(miniBatch)
    }

    override def train(miniBatch: MiniBatch, delta: Delta): Unit = {
      // Do nothing
    }

  }


  object If {

    sealed trait IfState[Data, Delta] {
      val data: Data
      val delta: Delta
      val count: Int
    }

    final case class Then[Data, Delta](data: Data, delta: Delta, count: Int) extends IfState[Data, Delta]

    final case class Else[Data, Delta](data: Data, delta: Delta, count: Int) extends IfState[Data, Delta]

  }

  import If._

  final case class If[Data0, Delta0](condition: Boolean, `then`: Differentiable.Aux[Data0, Delta0], `else`: Differentiable.Aux[Data0, Delta0]) extends Differentiable {

    override type Data = Data0
    override type Delta = Delta0

    private val cache = collection.mutable.HashMap.empty[MiniBatch, IfState[Data, Delta]]

    implicit override def monoid = `then`.monoid

    override def predict(miniBatch: MiniBatch): Data0 = {
      synchronized {
        cache.get(miniBatch)
      } match {
        case None =>
          if (condition.predict(miniBatch)) {
            val newData = `then`.predict(miniBatch)
            synchronized {
              cache.put(miniBatch, Then(newData, monoid.empty, 1))
            }
            newData
          } else {
            val newData = `else`.predict(miniBatch)
            synchronized {
              cache.put(miniBatch, Else(newData, monoid.empty, 1))
            }
            newData
          }
        case Some(Else(data, delta, count)) =>
          synchronized {
            cache.put(miniBatch, Else(data, delta, count + 1))
          }
          data
        case Some(Then(data, delta, count)) =>
          synchronized {
            cache.put(miniBatch, Then(data, delta, count + 1))
          }
          data
      }
    }

    override def train(miniBatch: MiniBatch, delta: Delta0): Unit = {

      synchronized {
        cache(miniBatch)
      } match {
        case Then(data, originalDelta, count) =>
          val newDelta = delta |+| originalDelta
          if (count == 1) {
            synchronized {
              cache.remove(miniBatch)
            }
            `then`.train(miniBatch, newDelta)
            condition.train(miniBatch, condition.monoid.empty)
          } else {
            synchronized {
              cache.put(miniBatch, Then(data, newDelta, count - 1))
            }
          }
        case Else(data, originalDelta, count) =>
          val newDelta = delta |+| originalDelta
          if (count == 1) {
            synchronized {
              cache.remove(miniBatch)
            }
            `else`.train(miniBatch, newDelta)
            condition.train(miniBatch, condition.monoid.empty)
          } else {
            synchronized {
              cache.put(miniBatch, Else(data, newDelta, count - 1))
            }
          }
      }
    }
  }

  def booleanOps(underlying: Boolean) = new BooleanOps[Any, Boolean] {
    override def `if`[A <: Any](`then`: A)(`else`: A) = {
      If(underlying, `then`, `else`)
    }
  }

  def doubleOps(underlying: Double) = new DoubleOps[Double] {
    override def unary_- = new Cached with DifferentiableDouble {

      override final def forward(miniBatch: MiniBatch): Data = {
        -underlying.predict(miniBatch)
      }

      override final def backward(miniBatch: MiniBatch, data: Data, delta: Delta): Unit = {
        underlying.train(miniBatch, -delta)
      }
    }.self

    override def -(rightHandSide: Double) = new Cached with DifferentiableDouble {
      override def forward(miniBatch: MiniBatch): scala.Double = {
        underlying.predict(miniBatch) - rightHandSide.predict(miniBatch)
      }

      override def backward(miniBatch: MiniBatch, data: scala.Double, delta: scala.Double): Unit = {
        underlying.train(miniBatch, delta)
        rightHandSide.train(miniBatch, -delta)
      }
    }.self

  }

}
