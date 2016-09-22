package com.thoughtworks.deepLearning

import cats.{Applicative, Eval, Monoid}

import scala.language.existentials
import scala.language.implicitConversions
import scala.language.higherKinds
import cats.implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import shapeless.{::, DepFn0, Generic, HNil}
import simulacrum.typeclass


object Differentiable {

  object Batch {
    type Aux[+Data0, -Delta0] = Batch {
      type Data <: Data0
      type Delta >: Delta0
    }

    case object HNilBatch extends Batch {
      override type Data = shapeless.HNil
      override type Delta = shapeless.HNil

      override def backward(delta: Delta): Unit = {
      }

      override def value: Data = shapeless.HNil
    }

    trait HConsBatch extends Batch {

      type HeadData
      type TailData <: shapeless.HList
      type HeadDelta
      type TailDelta <: shapeless.HList

      type Head = Batch.Aux[HeadData, HeadDelta]
      type Tail = Batch.Aux[TailData, TailDelta]

      override type Data = HeadData :: TailDelta
      override type Delta = HeadDelta :: TailDelta

    }

    trait DoubleBatch extends Batch {

      override type Data = Eval[scala.Double]

      override type Delta = Eval[scala.Double]

      // TODO: remove this monoid, use companion instead
      final def monoid: Monoid[Delta] = implicitly

    }


    trait BooleanBatch extends Batch {

      override type Data = Eval[scala.Boolean]

      override type Delta = Eval[scala.Boolean]

      final def monoid = new Monoid[Delta] {
        override def empty = Eval.now(false)

        override def combine(x: Eval[scala.Boolean], y: Eval[scala.Boolean]) = x.map2(y)(_ ^ _)
      }

    }

    trait Array2DBatch extends Batch {

      override type Data = Eval[INDArray]

      override type Delta = Eval[Option[INDArray]]

      final def monoid = new Monoid[Delta] {
        override def empty: Eval[Option[INDArray]] = Eval.now(None)

        override def combine(x: Delta, y: Delta): Delta = x.map2(y) {
          case (None, None) => None
          case (xDelta@Some(_), None) => xDelta
          case (None, yDelta@Some(_)) => yDelta
          case (Some(xDeltaValue), Some(yDeltaValue)) => Some(xDeltaValue add yDeltaValue)
        }
      }

    }

  }

  trait Batch {
    type Data
    type Delta

    def backward(delta: Delta): Unit

    def value: Data

  }

  type Aux[-Input0 <: Batch, +Output0 <: Batch.Aux[scala.Any, scala.Nothing]] = Differentiable {
    type Input >: Input0
    type Output <: Output0
  }

  final case class Id[Data0, Delta0]() extends Differentiable {
    outer =>
    type Input = Batch.Aux[Data0, Delta0]
    type Output = Batch.Aux[Data0, Delta0]

    override def forward(input: Input): Output = {
      input
    }
  }

  trait Cached extends Differentiable {

    private val cache = java.util.Collections.synchronizedMap(new java.util.IdentityHashMap[Input, Output with ReferenceCount](1))

    trait ReferenceCount extends Batch {
      private[Cached] var count: Int = 1

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

  import Batch._

  final case class Compose[A <: Batch, B <: Batch, C <: Batch](f: Differentiable.Aux[B, C], g: Differentiable.Aux[A, B]) extends Differentiable {
    override type Input = A
    override type Output = C

    override def forward(input: A): C = {
      f.forward(g.forward(input))
    }

  }

  object DifferentiableHCons {

    final case class Head[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList, TailDelta <: shapeless.HList]
    (
      differentiableHCons: Differentiable.Aux[Input0, Batch.Aux[shapeless.::[HeadData, TailData], shapeless.::[HeadDelta, TailDelta]]]
    )(implicit tailMonoid: Monoid[TailDelta]) extends Differentiable {
      override type Input = Input0

      final class Output(upstream: Batch.Aux[shapeless.::[HeadData, TailData], shapeless.::[HeadDelta, TailDelta]]) extends Batch {
        override def backward(delta: Delta): Unit = {
          upstream.backward(delta :: tailMonoid.empty)
        }

        override def value: Data = {
          upstream.value.head
        }

        override type Data = HeadData
        override type Delta = HeadDelta
      }

      override def forward(input: Input) = {
        new Output(differentiableHCons.forward(input))
      }
    }

    final case class Tail[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList, TailDelta <: shapeless.HList]
    (
      differentiableHCons: Differentiable.Aux[Input0, Batch.Aux[shapeless.::[HeadData, TailData], shapeless.::[HeadDelta, TailDelta]]]
    )(implicit headMonoid: Monoid[HeadDelta]) extends Differentiable {
      override type Input = Input0

      final class Output(upstream: Batch.Aux[shapeless.::[HeadData, TailData], shapeless.::[HeadDelta, TailDelta]]) extends Batch {
        override def backward(delta: Delta): Unit = {
          upstream.backward(headMonoid.empty :: delta)
        }

        override def value: Data = {
          upstream.value.tail
        }

        override type Data = TailData
        override type Delta = TailDelta
      }

      override def forward(input: Input) = {
        new Output(differentiableHCons.forward(input))
      }
    }

  }

  final case class DifferentiableHCons[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList, TailDelta <: shapeless.HList]
  (
    head: Differentiable.Aux[Input0, Batch.Aux[HeadData, HeadDelta]],
    tail: Differentiable.Aux[Input0, Batch.Aux[TailData, TailDelta]]
  ) extends Differentiable {
    override type Input = Input0

    final class Output(headBatch: Batch.Aux[HeadData, HeadDelta], tailBatch: Batch.Aux[TailData, TailDelta]) extends Batch {
      override def backward(delta: Delta): Unit = {
        headBatch.backward(delta.head)
        tailBatch.backward(delta.tail)
      }

      override def value: Data = {
        headBatch.value :: tailBatch.value
      }

      override type Data = shapeless.::[HeadData, TailData]
      override type Delta = shapeless.::[HeadDelta, TailDelta]
    }

    override def forward(input: Input) = {
      new Output(head.forward(input), tail.forward(input))
    }

  }


  final case class DifferentiableHNil[Input0 <: Batch]() extends Differentiable {
    override type Input = Input0
    override type Output = Batch.Aux[HNil, HNil]

    override def forward(input: Input): Output = HNilBatch
  }

  final case class DoubleLessThanDouble[Input0 <: Batch]
  (
    leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
    rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  ) extends Cached with Differentiable {

    final class Output(val input: Input0, upstream1: Batch.Aux[Eval[scala.Double], Eval[scala.Double]], upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with BooleanBatch {
      type Input >: Input0
      val value = upstream1.value.map2(upstream2.value)(_ < _).memoize

      override protected def cachedBackward(delta: Eval[scala.Boolean]): Unit = {
        upstream1.backward(Eval.now(0.0))
        upstream2.backward(Eval.now(0.0))
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): Output = {
      new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }
  }

  final case class DoubleAddDouble[Input0 <: Batch]
  (
    leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
    rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  ) extends Cached with Differentiable {

    final class Output(val input: Input0, upstream1: Batch.Aux[Eval[scala.Double], Eval[scala.Double]], upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with DoubleBatch {
      type Input >: Input0
      val value = upstream1.value.map2(upstream2.value)(_ + _)

      override protected def cachedBackward(delta: Eval[scala.Double]): Unit = {
        upstream1.backward(delta)
        upstream2.backward(delta)
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): Output = {
      new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }

  }

  final case class DoubleSubtractDouble[Input0 <: Batch]
  (
    leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
    rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  ) extends Cached with Differentiable {

    final class Output(val input: Input0, upstream1: Batch.Aux[Eval[scala.Double], Eval[scala.Double]], upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with DoubleBatch {
      type Input >: Input0
      val value = upstream1.value.map2(upstream2.value)(_ - _)

      override protected def cachedBackward(delta: Eval[scala.Double]): Unit = {
        upstream1.backward(delta)
        upstream2.backward(delta.map(-_))
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): Output = {
      new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }

  }

  final case class DoubleMultiplyDouble[Input0 <: Batch]
  (
    leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
    rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  ) extends Cached with Differentiable {

    final class Output(val input: Input0, upstream1: Batch.Aux[Eval[scala.Double], Eval[scala.Double]], upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with DoubleBatch {
      type Input >: Input0
      val value = upstream1.value.map2(upstream2.value)(_ * _)

      override protected def cachedBackward(delta: Eval[scala.Double]): Unit = {
        val a = upstream1.value
        val b = upstream2.value
        upstream1.backward(delta.map2(b)(_ * _))
        upstream2.backward(delta.map2(a)(_ * _))
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): Output = {
      new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }

  }

  final case class DoubleReciprocal[Input0 <: Batch](differentiableDouble: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]) extends Cached with Differentiable {

    final class Output(val input: Input0, upstream: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with DoubleBatch {
      type Input >: Input0
      val value = upstream.value.map(1.0 / _)

      override protected def cachedBackward(delta: Eval[scala.Double]): Unit = {
        val a = upstream.value
        upstream.backward(delta.map2(a) {
          (outputDeltaValue: scala.Double, aValue: scala.Double) =>
            -outputDeltaValue / (aValue * aValue)
        })
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): Output = {
      val upstream = differentiableDouble.forward(input)
      new Output(input, upstream)
    }
  }

  final case class DoubleNegative[Input0 <: Batch](differentiableDouble: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]) extends Cached with Differentiable {

    final class Output(val input: Input0, upstream: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with DoubleBatch {
      type Input >: Input0
      val value = upstream.value.map(-_)

      override protected def cachedBackward(delta: Eval[scala.Double]): Unit = {
        upstream.backward(delta.map(-_))
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): Output = {
      val upstream = differentiableDouble.forward(input)
      new Output(input, upstream)
    }
  }

  final case class DoubleLiteral[Input0 <: Batch](rawValue: scala.Double) extends Differentiable with DoubleBatch {
    override type Input = Input0
    override type Output = DoubleLiteral[Input0]

    override def forward(any: Input) = this

    override def backward(delta: Delta): Unit = {}

    override def value = Eval.now(rawValue)
  }

  final case class DoubleWeight[Input0 <: Batch](var rawValue: scala.Double)(implicit learningRate: LearningRate) extends Differentiable with DoubleBatch {
    override type Input = Input0
    override type Output = DoubleWeight[Input0]

    override def forward(any: Input) = this

    override def backward(delta: Delta): Unit = {
      rawValue -= delta.value * learningRate()
    }

    override def value = Eval.now(rawValue)

  }

  final case class BooleanLiteral[Input0 <: Batch](rawValue: scala.Boolean) extends Differentiable with BooleanBatch {
    override type Input = Input0
    override type Output = BooleanLiteral[Input0]

    override def forward(any: Input) = this

    override def backward(delta: Delta): Unit = {}

    override def value = Eval.now(rawValue)
  }

  final case class BooleanWeight[Input0 <: Batch](var rawValue: scala.Boolean) extends Differentiable with BooleanBatch {
    override type Input = Input0
    override type Output = BooleanWeight[Input0]

    override def forward(any: Input) = this

    override def backward(delta: Delta): Unit = {
      rawValue ^= delta.value
    }

    override def value = Eval.now(rawValue)

  }

  final case class Array2DLiteral[Input0 <: Batch](rawValue: INDArray) extends Differentiable with Array2DBatch {
    override type Input = Input0
    override type Output = Array2DLiteral[Input0]

    override def value = Eval.now(rawValue)

    override def forward(any: Input) = this

    override def backward(delta: Delta): Unit = {}
  }

  object Array2DLiteral {
    def apply[Input <: Batch](nativeArray: Array[Array[scala.Double]]): Array2DLiteral[Input] = {
      new Array2DLiteral[Input](nativeArray.toNDArray)
    }
  }

  final case class Array2DWeight[Input0 <: Batch](var rawValue: INDArray)(implicit learningRate: LearningRate) extends Differentiable with Array2DBatch {
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
    def randn[Input <: Batch](numberOfRows: Int, numberOfColumns: Int)(implicit learningRate: LearningRate): Array2DWeight[Input] = {
      new Array2DWeight[Input](Nd4j.randn(numberOfRows, numberOfColumns))
    }

    def zeros[Input <: Batch](numberOfRows: Int, numberOfColumns: Int)(implicit learningRate: LearningRate): Array2DWeight[Input] = {
      new Array2DWeight[Input](Nd4j.zeros(numberOfRows, numberOfColumns))
    }

    def apply[Input <: Batch](nativeArray: Array[Array[scala.Double]])(implicit learningRate: LearningRate): Array2DWeight[Input] = {
      new Array2DWeight[Input](nativeArray.toNDArray)
    }

  }

  final case class Array2DOps[Input0 <: Batch](generic: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Differentiable {
    override type Output = generic.Output
    override type Input = Input0

    override def forward(input: Input0): generic.Output = {
      generic.forward(input)
    }
  }

  final case class Dot[Input0 <: Batch]
  (
    leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
    rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]
  ) extends Differentiable with Cached {

    final class Output(val input: Input0, upstream1: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]], upstream2: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with Array2DBatch {
      type Input >: Input0
      val value = upstream1.value.map2(upstream2.value)(_ dot _).memoize

      override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
        val b = upstream2.value
        upstream1.backward(outputDelta.flatMap[Option[INDArray]] {
          case None => Eval.now(None)
          case Some(outputDeltaValue) =>
            b.map {
              bData =>
                Some(outputDeltaValue.dot(bData.T))
            }
        }.memoize)
        val a = upstream1.value
        upstream2.backward(outputDelta.flatMap[Option[INDArray]] {
          case None => Eval.now(None)
          case Some(outputDeltaValue) =>
            a.map {
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


  private def sumAs(outputDeltaValue: INDArray, shape: Array[Int]) = shape match {
    case Array(1, 1) => outputDeltaValue.sum(0, 1)
    case Array(_, 1) => outputDeltaValue.sum(1)
    case Array(1, _) => outputDeltaValue.sum(0)
    case Array(_, _) => outputDeltaValue
  }

  final case class Array2DMultiplyArray2D[Input0 <: Batch]
  (
    leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
    rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]
  ) extends Differentiable with Cached {

    final class Output(val input: Input0, upstream1: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]], upstream2: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with Array2DBatch {
      type Input >: Input0
      val value = upstream1.value.map2(upstream2.value)(_ * _).memoize

      override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
        val a = upstream1.value
        val b = upstream2.value
        upstream1.backward(outputDelta.flatMap[Option[INDArray]] {
          case None => Eval.now(None)
          case Some(outputDeltaValue) =>
            a.map2(b) { (aData, bData) =>
              Some(sumAs(bData.broadcast(outputDeltaValue.shape(): _*) * outputDeltaValue, aData.shape()))
            }
        }.memoize)
        upstream2.backward(outputDelta.flatMap[Option[INDArray]] {
          case None => Eval.now(None)
          case Some(outputDeltaValue) =>
            a.map2(b) { (aData, bData) =>
              Some(sumAs(aData.broadcast(outputDeltaValue.shape(): _*) * outputDeltaValue, bData.shape()))
            }
        }.memoize)
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): Output = {
      new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }
  }

  final case class Array2DAddArray2D[Input0 <: Batch]
  (
    leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
    rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]
  ) extends Differentiable with Cached {

    final class Output(val input: Input0, upstream1: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]], upstream2: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with Array2DBatch {
      type Input >: Input0
      val value = upstream1.value.map2(upstream2.value)(_ + _).memoize

      override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
        val sumAsOriginalShape = { (outputDeltaOption: Option[INDArray], aValue: INDArray) =>
          outputDeltaOption.map(sumAs(_, aValue.shape()))
        }
        upstream1.backward(outputDelta.map2(upstream1.value)(sumAsOriginalShape))
        upstream2.backward(outputDelta.map2(upstream2.value)(sumAsOriginalShape))
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): Output = {
      new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }
  }

  final case class Array2DMultiplyDouble[Input0 <: Batch]
  (
    leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
    rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  ) extends Differentiable with Cached {

    final class Output(val input: Input0, upstream1: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]], upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with Array2DBatch {
      type Input >: Input0
      val value = upstream1.value.map2(upstream2.value)(_ * _).memoize

      override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {

        val a = upstream1.value
        val b = upstream2.value

        val aDelta = outputDelta.flatMap[Option[INDArray]] {
          case None => Eval.now(None)
          case Some(outputDeltaValue) =>
            b.map {
              bData: scala.Double =>
                Some(outputDeltaValue * bData)
            }
        }.memoize
        upstream1.backward(aDelta)
        val bDelta = outputDelta.flatMap[scala.Double] {
          case None => Eval.now(0.0)
          case Some(outputDeltaValue) =>
            a.map {
              aData: INDArray =>
                (aData * outputDeltaValue).sumT
            }
        }.memoize
        upstream2.backward(bDelta)
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): Output = {
      new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }
  }

  final case class Array2DMaxDouble[Input0 <: Batch]
  (
    leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
    rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  ) extends Differentiable with Cached {

    final class Output(val input: Input0, upstream1: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]], upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with Array2DBatch {
      type Input >: Input0
      val value = upstream1.value.map2(upstream2.value)(Transforms.max).memoize

      override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
        val a = upstream1.value
        val b = upstream2.value
        upstream1.backward(outputDelta.flatMap[Option[INDArray]] {
          case None => Eval.now(None)
          case Some(outputDeltaValue) =>
            Applicative[Eval].map2(a, b) {
              (aData: INDArray, bData: scala.Double) =>
                Some((aData gt bData) * outputDeltaValue)
            }
        })
        upstream2.backward(outputDelta.flatMap[scala.Double] {
          case None => Eval.now(0)
          case Some(outputDeltaValue) =>
            Applicative[Eval].map2(a, b) {
              (aData: INDArray, bData: scala.Double) =>
                ((aData lt bData) * outputDeltaValue).sumT
            }
        })
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): Output = {
      new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }
  }

  final case class Array2DAddDouble[Input0 <: Batch]
  (
    leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
    rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  ) extends Differentiable with Cached {

    final class Output(val input: Input0, upstream1: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]], upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with Array2DBatch {
      type Input >: Input0
      val value = upstream1.value.map2(upstream2.value)(_ + _).memoize

      override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
        upstream1.backward(outputDelta)
        upstream2.backward(outputDelta.map {
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

  final case class Array2DReciprocal[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Cached with Differentiable {

    final class Output(val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with Array2DBatch {
      type Input >: Input0
      val value = upstream.value.map(_ rdiv 1.0).memoize


      override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
        val upstreamValue = upstream.value
        upstream.backward(outputDelta.flatMap[Option[INDArray]] {
          case None => Eval.now(None)
          case Some(outputDeltaValue) =>
            upstreamValue.map {
              aValue: INDArray =>
                Some(-outputDeltaValue / (aValue * aValue))
            }
        }.memoize)
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): Output = {
      val upstream = differentiableArray2D.forward(input)
      new Output(input, upstream)
    }
  }


  final case class ReduceSum[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Cached with Differentiable {

    final class Output(val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with DoubleBatch {
      type Input >: Input0
      val value = upstream.value.map(_.sumT).memoize

      override protected def cachedBackward(outputDelta: Eval[scala.Double]): Unit = {
        upstream.backward(outputDelta.map2(upstream.value) { (outputDeltaValue, aValue) =>
          if (outputDeltaValue == 0) {
            None
          } else {
            Some(Nd4j.valueArrayOf(aValue.shape(), outputDeltaValue))
          }
        }.memoize)
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): Output = {
      val upstream = differentiableArray2D.forward(input)
      new Output(input, upstream)
    }
  }

  final case class Sum[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]], dimensions: Seq[Int]) extends Cached with Differentiable {

    final class Output(val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with Array2DBatch {
      type Input >: Input0
      val value = upstream.value.map(_.sum(dimensions: _*)).memoize

      override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
        val a = upstream.value
        upstream.backward(
          outputDelta.flatMap[Option[INDArray]] {
            case None => Eval.now(None)
            case Some(outputDeltaValue) =>
              a.map {
                aValue =>
                  Some(outputDeltaValue.broadcast(aValue.shape(): _*))
              }
          }.memoize)
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): Output = {
      val upstream = differentiableArray2D.forward(input)
      new Output(input, upstream)
    }
  }

  final case class Array2DNegative[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Cached with Differentiable {

    final class Output(val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with Array2DBatch {
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
      val upstream = differentiableArray2D.forward(input)
      new Output(input, upstream)
    }
  }

  final case class Array2DLog[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Cached with Differentiable {

    final class Output(val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with Array2DBatch {
      type Input >: Input0
      val value = upstream.value.map(Transforms.log).memoize

      override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
        val a = upstream.value
        upstream.backward(outputDelta.flatMap[Option[INDArray]] {
          case None => Eval.now(None)
          case Some(outputDeltaValue) => a.map[Option[INDArray]] {
            aData: INDArray =>
              Some(outputDeltaValue / aData)
          }
        }.memoize)
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): Output = {
      val upstream = differentiableArray2D.forward(input)
      new Output(input, upstream)
    }
  }

  final case class Array2DExp[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Cached with Differentiable {

    final class Output(val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with Array2DBatch {
      type Input >: Input0
      val value = upstream.value.map(Transforms.exp).memoize

      override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
        upstream.backward(outputDelta.flatMap {
          case None => Eval.now(None)
          case Some(outputDeltaValue) => value.map {
            outputValue: INDArray =>
              Some(outputValue * outputDeltaValue)
          }
        }.memoize)
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): Output = {
      val upstream = differentiableArray2D.forward(input)
      new Output(input, upstream)
    }
  }

  final case class If[Input0 <: Batch, Output0 <: Batch](condition: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]],
                                                         `then`: Differentiable.Aux[Input0, Output0],
                                                         `else`: Differentiable.Aux[Input0, Output0])
    extends Differentiable {
    override type Input = Input0
    override type Output = Output0

    override def forward(input: Input0): Output0 = {
      val conditionForwardPass = condition.forward(input)
      val output = if (conditionForwardPass.value.value) {
        `then`.forward(input)
      } else {
        `else`.forward(input)
      }
      conditionForwardPass.backward(Eval.now(false))
      output
    }
  }


  final case class Not[Input0 <: Batch](differentiableBoolean: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]]) extends Cached with Differentiable {

    final class Output(val input: Input0, upstream: Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]) extends ReferenceCount with BooleanBatch {
      type Input >: Input0
      val value = upstream.value.map(!_)

      override protected def cachedBackward(delta: Eval[scala.Boolean]): Unit = {
        upstream.backward(delta.map(!_))
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): Output = {
      val upstream = differentiableBoolean.forward(input)
      new Output(input, upstream)
    }
  }

  trait LearningRate {
    def apply(): scala.Double
  }

  object SymbolicDsl {
    type Aux[Input0 <: Batch] = SymbolicDsl {
      type Input = Input0
    }


    def apply[Input0 <: Batch](implicit learningRate0: LearningRate): SymbolicDsl.Aux[Input0] = new SymbolicDsl {
      override protected def learningRate = learningRate0

      override protected type Input = Input0
    }

  }

  trait SymbolicDsl extends Dsl {
    implicit protected def learningRate: LearningRate

    trait Any {
      type OutputData <: scala.Any
      type OutputDelta >: scala.Nothing
      val underlying: Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]]
    }

    protected type Input <: Batch

    private[SymbolicDsl] object Companion {
      type Aux[Ast <: Any, Data, Delta] = Companion[Ast] {
        type OutputData = Data
        type OutputDelta = Delta
      }
    }

    trait Companion[Ast <: Any] extends Dsl.Lifter {
      _: (_ => Ast) =>
      type LiftTo = Ast
      type OutputData
      type OutputDelta

      def monoid: Monoid[OutputDelta]

      def toDifferentiable(ast: Ast): Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]]

      def toAst(generic: Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]]): Ast
    }

    protected object HListCompanion {
      type Aux[Ast <: HList, Data, Delta] = HListCompanion[Ast] {
        type OutputData = Data
        type OutputDelta = Delta
      }
    }

    trait HListCompanion[Ast <: HList] extends Companion[Ast] {
      _: (_ => Ast) =>

      override type OutputData <: shapeless.HList
      override type OutputDelta <: shapeless.HList
      override type LiftFrom <: shapeless.HList
    }

    trait HList extends HListApi with Any {

      override type OutputData <: shapeless.HList
      override type OutputDelta <: shapeless.HList

      override def ::[Head <: Any, Tail >: this.type <: HList](head: Head)(implicit headCompanion: Companion[Head], tailCompanion: HListCompanion[Tail]): Head :: Tail = {
        SymbolicDsl.this.::[Head, Tail].toAst(DifferentiableHCons(headCompanion.toDifferentiable(head), tailCompanion.toDifferentiable(this)))
      }
    }

    sealed trait HNil extends HList {
      override type OutputData = shapeless.HNil
      override type OutputDelta = shapeless.HNil
    }

    case object HNil extends HNil with HListCompanion[HNil] with (shapeless.HNil => HNil) {
      override def toAst(generic: Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) = this

      override val underlying = DifferentiableHNil[Input]()

      override def toDifferentiable(ast: HNil) = ast.underlying

      override def monoid = new Monoid[OutputDelta] {
        override def empty = shapeless.HNil

        override def combine(x: OutputDelta, y: OutputDelta) = shapeless.HNil
      }

      override type LiftFrom = shapeless.HNil

      override def apply(value: shapeless.HNil) = this

      override def weight(initialValue: shapeless.HNil) = this

    }

    protected final case class HCons[+Head <: Any, +Tail <: HList, HeadData, HeadDelta, TailData <: shapeless.HList, TailDelta <: shapeless.HList]
    (underlying: Differentiable.Aux[Input, Batch.Aux[shapeless.::[HeadData, TailData], shapeless.::[HeadDelta, TailDelta]]])
    (
      implicit headCompanion: Companion.Aux[Head, HeadData, HeadDelta],
      tailCompanion: HListCompanion.Aux[Tail, TailData, TailDelta]
    )
      extends (Head :: Tail) {

      override type OutputData = shapeless.::[HeadData, TailData]
      override type OutputDelta = shapeless.::[HeadDelta, TailDelta]

      def head: Head = {
        headCompanion.toAst(DifferentiableHCons.Head[Input, HeadData, HeadDelta, TailData, TailDelta](underlying)(tailCompanion.monoid))
      }

      def tail: Tail = {
        tailCompanion.toAst(DifferentiableHCons.Tail[Input, HeadData, HeadDelta, TailData, TailDelta](underlying)(headCompanion.monoid))
      }

    }

    sealed trait ::[+Head <: Any, +Tail <: HList] extends HList with HConsApi[Head, Tail]

    abstract class HConsCompanion[Head <: Any, Tail <: HList, HeadCompanion <: Companion[Head], TailCompanion <: HListCompanion[Tail]]
    ()
    (implicit val headCompanion: HeadCompanion, val tailCompanion: TailCompanion)
      extends HListCompanion[Head :: Tail] with (shapeless.::[HeadCompanion#LiftFrom, TailCompanion#LiftFrom] => (Head :: Tail)) {

      override type OutputData = shapeless.::[headCompanion.OutputData, tailCompanion.OutputData]
      override type OutputDelta = shapeless.::[headCompanion.OutputDelta, tailCompanion.OutputDelta]

      override type LiftFrom = shapeless.::[headCompanion.LiftFrom, tailCompanion.LiftFrom]

      override def apply(value: LiftFrom): Head :: Tail = {
        headCompanion(value.head) :: tailCompanion(value.tail)
      }

      override def weight(initialValue: LiftFrom): Head :: Tail = {
        headCompanion.weight(initialValue.head) :: tailCompanion.weight(initialValue.tail)
      }


      override def monoid = new Monoid[OutputDelta] {
        override def empty: OutputDelta = headCompanion.monoid.empty :: tailCompanion.monoid.empty

        override def combine(x: OutputDelta, y: OutputDelta) = {
          implicit val headMonoid = headCompanion.monoid
          implicit val tailMonoid = tailCompanion.monoid
          (x.head |+| y.head) :: (x.tail |+| y.tail)
        }
      }

      override def toAst(generic: Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) = {
        new HCons(generic)(headCompanion, tailCompanion)
      }

      override def toDifferentiable(ast: Head :: Tail): Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = {
        ast.asInstanceOf[(Head :: Tail) {
          type OutputData = HConsCompanion.this.OutputData
          type OutputDelta = HConsCompanion.this.OutputDelta
        }].underlying
      }
    }

    override final def ::[Head <: Any, Tail <: HList](implicit headCompanion: Companion[Head], tailCompanion: HListCompanion[Tail]): HConsCompanion[Head, Tail, headCompanion.type, tailCompanion.type] = {
      new HConsCompanion[Head, Tail, headCompanion.type, tailCompanion.type]()(headCompanion, tailCompanion) {}
    }

    object Boolean extends Companion[Boolean] with (scala.Boolean => Boolean) {

      type LiftFrom = scala.Boolean

      def weight(initialValue: LiftFrom): LiftTo = {
        toAst(BooleanWeight(initialValue))
      }

      def apply(value: LiftFrom): LiftTo = {
        toAst(BooleanLiteral(value))
      }

      override type OutputData = Eval[scala.Boolean]
      override type OutputDelta = Eval[scala.Boolean]

      override def monoid = new Monoid[Eval[scala.Boolean]] {
        override def empty = Eval.now(false)

        override def combine(x: Eval[scala.Boolean], y: Eval[scala.Boolean]) = x.map2(y)(_ ^ _)
      }

      override def toDifferentiable(ast: Boolean) = {
        ast.underlying
      }

      override def toAst(generic: Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) = Boolean(generic)
    }

    final case class Boolean(underlying: Differentiable.Aux[Input, Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]]) extends BooleanApi with Any {
      override type OutputData = Eval[scala.Boolean]
      override type OutputDelta = Eval[scala.Boolean]

      override def unary_! : Boolean = {
        Boolean.toAst(Not(underlying))
      }

      override def `if`[Ast <: Any](`then`: Ast)(`else`: Ast)(implicit companion: Companion[Ast]): Ast = {
        companion.toAst(If(underlying, companion.toDifferentiable(`then`), companion.toDifferentiable(`else`)))
      }

    }

    object Double extends Companion[Double] with (scala.Double => Double) {

      type LiftFrom = scala.Double

      def weight(initialValue: LiftFrom): LiftTo = toAst(DoubleWeight(initialValue))

      def apply(value: LiftFrom): LiftTo = toAst(DoubleLiteral(value))

      override type OutputData = Eval[scala.Double]
      override type OutputDelta = Eval[scala.Double]

      override def monoid = implicitly

      override def toDifferentiable(ast: Double) = ast.underlying

      override def toAst(generic: Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) = Double(generic)
    }

    final case class Double(underlying: Differentiable.Aux[Input, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]) extends DoubleApi with Any {
      override type OutputData = Eval[scala.Double]
      override type OutputDelta = Eval[scala.Double]

      override def unary_- : Double = {
        Double.toAst(DoubleNegative(underlying))
      }

      override def +(rightHandSide: Double): Double = {
        Double.toAst(DoubleAddDouble(underlying, rightHandSide.underlying))
      }

      override def /(rightHandSide: Double): Double = {
        Double.toAst(DoubleMultiplyDouble(underlying, DoubleReciprocal(rightHandSide.underlying)))
      }

      override def /(rightHandSide: Array2D): Array2D = {
        Array2D.toAst(Array2DMultiplyDouble(Array2DReciprocal(rightHandSide.underlying), underlying))
      }

      override def *(rightHandSide: Double): Double = {
        Double.toAst(DoubleMultiplyDouble(underlying, rightHandSide.underlying))
      }

      override def <(rightHandSide: Double): Boolean = {
        Boolean.toAst(DoubleLessThanDouble(underlying, rightHandSide.underlying))
      }

    }

    object Array2D extends Companion[Array2D] with Array2DCompanionApi {

      def weight(initialValue: LiftFrom): LiftTo = toAst(Array2DWeight(initialValue))

      def apply(value: LiftFrom): LiftTo = toAst(Array2DLiteral(value))

      override type OutputData = Eval[INDArray]
      override type OutputDelta = Eval[Option[INDArray]]

      override def monoid = new Monoid[OutputDelta] {
        override def empty = Eval.now(None)

        override def combine(x: OutputDelta, y: OutputDelta): OutputDelta = x.map2(y) {
          case (None, None) => None
          case (xDelta@Some(_), None) => xDelta
          case (None, yDelta@Some(_)) => yDelta
          case (Some(xDeltaValue), Some(yDeltaValue)) => Some(xDeltaValue add yDeltaValue)
        }

      }

      override def toDifferentiable(ast: Array2D) = ast.underlying

      override def toAst(generic: Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) = Array2D(generic)

      override def randn(numberOfRows: Int, numberOfColumns: Int): Array2D = {
        Array2D.toAst(Array2DWeight.randn(numberOfRows, numberOfColumns))
      }

      override def zeros(numberOfRows: Int, numberOfColumns: Int): Array2D = {
        Array2D.toAst(Array2DWeight.zeros(numberOfRows, numberOfColumns))
      }

    }

    final case class Array2D(underlying: Differentiable.Aux[Input, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Array2DApi with Any {
      override type OutputData = Eval[INDArray]
      override type OutputDelta = Eval[Option[INDArray]]

      override def dot(rightHandSide: Array2D) = {
        Array2D.toAst(Dot(underlying, rightHandSide.underlying))
      }

      override def +(rightHandSide: Array2D) = {
        Array2D.toAst(Array2DAddArray2D(underlying, rightHandSide.underlying))
      }

      override def +(rightHandSide: Double) = {
        Array2D.toAst(Array2DAddDouble(underlying, rightHandSide.underlying))
      }

      override def /(rightHandSide: Array2D) = {
        Array2D.toAst(Array2DMultiplyArray2D(underlying, Array2DReciprocal(rightHandSide.underlying)))
      }

      override def /(rightHandSide: Double) = {
        Array2D.toAst(Array2DMultiplyDouble(underlying, DoubleReciprocal(rightHandSide.underlying)))
      }

      override def *(rightHandSide: Array2D): Array2D = {
        Array2D.toAst(Array2DMultiplyArray2D(underlying, rightHandSide.underlying))
      }

      override def *(rightHandSide: Double): Array2D = {
        Array2D.toAst(Array2DMultiplyDouble(underlying, rightHandSide.underlying))
      }

      override def unary_- : Array2D = {
        Array2D.toAst(Array2DNegative(underlying))
      }

      override def reduceSum: Double = {
        Double.toAst(ReduceSum(underlying))
      }

      override def sum(dimensions: Int*): Array2D = {
        Array2D.toAst(Sum(underlying, dimensions))
      }
    }

    override def max(leftHandSide: Array2D, rightHandSide: Double): Array2D = {
      Array2D.toAst(Array2DMaxDouble(leftHandSide.underlying, rightHandSide.underlying))
    }

    override def exp(array: Array2D): Array2D = {
      Array2D.toAst(Array2DExp(array.underlying))
    }

    override def log(array: Array2D): Array2D = {
      Array2D.toAst(Array2DLog(array.underlying))
    }
  }


  trait SymbolicInput {
    type Ast[D <: SymbolicDsl] <: D#Any
    type OutputData
    type OutputDelta

    val dsl: SymbolicDsl.Aux[Batch.Aux[OutputData, OutputDelta]]

    val ast: Ast[dsl.type]

    def companion(anotherDsl: SymbolicDsl): anotherDsl.Companion[Ast[anotherDsl.type]] {
      type OutputData = SymbolicInput.this.OutputData
      type OutputDelta = SymbolicInput.this.OutputDelta
    }
  }

  object SymbolicInput {

    implicit def array2DInput(implicit learningRate: LearningRate) = new SymbolicInput {
      override type OutputData = Eval[INDArray]
      override type OutputDelta = Eval[Option[INDArray]]
      override type Ast[D <: SymbolicDsl] = D#Array2D

      override val dsl = SymbolicDsl[Batch.Aux[OutputData, OutputDelta]]

      override def companion(anotherDsl: SymbolicDsl) = anotherDsl.Array2D

      override val ast = dsl.Array2D(Id[OutputData, OutputDelta]())
    }

    implicit def doubleInput(implicit learningRate: LearningRate) = new SymbolicInput {
      override type OutputData = Eval[scala.Double]
      override type OutputDelta = Eval[scala.Double]
      override type Ast[D <: SymbolicDsl] = D#Double

      override val dsl = SymbolicDsl[Batch.Aux[OutputData, OutputDelta]]

      override def companion(anotherDsl: SymbolicDsl) = anotherDsl.Double

      override val ast = dsl.Double(Id[OutputData, OutputDelta]())
    }

    implicit def booleanInput(implicit learningRate: LearningRate) = new SymbolicInput {
      override type OutputData = Eval[scala.Boolean]
      override type OutputDelta = Eval[scala.Boolean]
      override type Ast[D <: SymbolicDsl] = D#Boolean

      override val dsl = SymbolicDsl[Batch.Aux[OutputData, OutputDelta]]

      override def companion(anotherDsl: SymbolicDsl) = anotherDsl.Boolean

      override val ast = dsl.Boolean(Id[OutputData, OutputDelta]())
    }

    implicit def hnilInput(implicit learningRate: LearningRate) = new SymbolicInput {
      override type OutputData = shapeless.HNil
      override type OutputDelta = shapeless.HNil
      override type Ast[D <: SymbolicDsl] = D#HNil

      override val dsl = SymbolicDsl[Batch.Aux[OutputData, OutputDelta]]

      override def companion(anotherDsl: SymbolicDsl) = anotherDsl.HNil

      override val ast = dsl.HNil
    }

    implicit def hconsInput[HeadData, HeadDelta, TailData <: shapeless.HList, TailDelta <: shapeless.HList]
    (implicit
     learningRate: LearningRate,
     headInput: SymbolicInput {
       type OutputData = HeadData
       type OutputDelta = HeadDelta
     },
     tailInput: SymbolicInput {
       type OutputData = TailData
       type OutputDelta = TailDelta

       type Ast[D <: SymbolicDsl] <: D#HList

       def companion(anotherDsl: SymbolicDsl): anotherDsl.HListCompanion[Ast[anotherDsl.type]] {
         type OutputData = TailData
         type OutputDelta = TailDelta
       }

     }
    ) = new SymbolicInput {
      override type OutputData = shapeless.::[HeadData, TailData]
      override type OutputDelta = shapeless.::[HeadDelta, TailDelta]
      override type Ast[D <: SymbolicDsl] = D# ::[headInput.Ast[D], tailInput.Ast[D]]

      override val dsl = SymbolicDsl[Batch.Aux[OutputData, OutputDelta]]

      override def companion(anotherDsl: SymbolicDsl) = {
        anotherDsl.::[headInput.Ast[anotherDsl.type], tailInput.Ast[anotherDsl.type]](headInput.companion(anotherDsl), tailInput.companion(anotherDsl))
      }

      override val ast: Ast[dsl.type] = {
        val id: Differentiable.Aux[Batch.Aux[OutputData, OutputDelta], Batch.Aux[OutputData, OutputDelta]] = Id[shapeless.::[HeadData, TailData], shapeless.::[HeadDelta, TailDelta]]()
        val headCompanion = headInput.companion(dsl)
        val tailCompanion = tailInput.companion(dsl)
        type Head = headInput.Ast[dsl.type]
        type Tail = tailInput.Ast[dsl.type]
        val companion: dsl.HConsCompanion[Head, Tail, headCompanion.type, tailCompanion.type] = dsl.::[Head, Tail](headCompanion, tailCompanion)
        companion.toAst(id)
      }
    }

  }

}

trait Differentiable {

  import Differentiable._

  type Input <: Batch

  type Output <: Batch.Aux[scala.Any, scala.Nothing]

  def forward(input: Input): Output

}