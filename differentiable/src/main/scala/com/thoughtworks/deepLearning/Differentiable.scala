package com.thoughtworks.deepLearning

import cats._

import scala.language.existentials
import scala.language.implicitConversions
import scala.language.higherKinds
import cats.implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._

import scala.annotation.{elidable, tailrec}
import shapeless._
import shapeless.ops.hlist.Length
import shapeless.ops.nat.ToInt

object Differentiable {

  type Aux[-Input0 <: Batch, +Output0 <: Batch.Aux[scala.Any, scala.Nothing]] =
    Differentiable {
      type Input >: Input0
      type Output <: Output0
    }

  object Batch {
    type Aux[+Data0, -Delta0] = Batch {
      type Data <: Data0
      type Delta >: Delta0
    }

    private[Differentiable] sealed trait DoubleBatch extends Batch {

      override type Data = Eval[scala.Double]

      override type Delta = Eval[scala.Double]

      final def monoid: Monoid[Delta] = implicitly

    }

    private[Differentiable] sealed trait BooleanBatch extends Batch {

      override type Data = Eval[scala.Boolean]

      override type Delta = Eval[scala.Boolean]

      final def monoid = new Monoid[Delta] {
        override def empty = Eval.now(false)

        override def combine(x: Eval[scala.Boolean], y: Eval[scala.Boolean]) = x.map2(y)(_ ^ _)
      }

    }

    private[Differentiable] sealed trait Array2DBatch extends Batch {

      override type Data = Eval[INDArray]

      override type Delta = Eval[INDArray]

      protected final def semigroup = new Semigroup[Delta] {
        override def combine(x: Delta, y: Delta): Delta = x.map2(y)(_ + _)
      }

    }

  }

  trait Batch extends AutoCloseable {
    type Data
    type Delta

    def backward(delta: Delta): Unit

    def value: Data
  }

  final case class Id[Data0, Delta0]() extends Differentiable { outer =>
    type Input = Batch.Aux[Data0, Delta0]
    type Output = Batch.Aux[Data0, Delta0]

    override def forward(input: Input): Output = {
      input
    }
  }

  private[Differentiable] sealed trait Cached extends Differentiable {

    private val cache =
      java.util.Collections.synchronizedMap(new java.util.IdentityHashMap[Input, SharedBatch](1))

    private[Cached] sealed trait ReferenceCount extends Batch { this: SharedBatch =>

      type Output = Batch.Aux[Data, Delta]

      @elidable(elidable.ASSERTION)
      private[Cached] def checked: Output = new Batch {
        type Delta = ReferenceCount.this.Delta
        type Data = ReferenceCount.this.Data

        override final def backward(delta: Delta) = ReferenceCount.this.backward(delta)

        def value = ReferenceCount.this.value

        private var closed = false

        override final def close(): Unit = {
          ReferenceCount.this.synchronized {
            if (closed) {
              throw new IllegalStateException("close() method must be called once and only once.")
            } else {
              closed = true
            }
          }
          ReferenceCount.this.close()
        }
      }

      private[Cached] var count: Int = 1

      private[Cached] def cachedBackward(): Unit

      protected def cachedClose(): Unit

      def input: Input

      override final def close(): Unit = {
        val newCount = synchronized {
          count -= 1
          count
        }
        if (newCount == 0) {
          cache.remove(input)
          cachedBackward()
          cachedClose()
        }
      }

    }

    private[Differentiable] sealed trait MonoidBatch extends ReferenceCount { this: SharedBatch =>

      private var currentDelta: Delta = monoid.empty

      protected def cachedBackward(delta: Delta): Unit

      implicit protected def monoid: Monoid[Delta]

      override private[Cached] final def cachedBackward(): Unit = {
        cachedBackward(currentDelta)
      }

      override final def backward(delta: Delta): Unit = {
        synchronized {
          currentDelta = currentDelta |+| delta
        }
      }
    }

    private[Differentiable] sealed trait SemigroupBatch extends ReferenceCount { this: SharedBatch =>

      private var currentDelta: Option[Delta] = None

      protected def cachedBackward(delta: Delta): Unit

      implicit protected def semigroup: Semigroup[Delta]

      override private[Cached] final def cachedBackward(): Unit = {
        currentDelta.foreach(cachedBackward)
      }

      override final def backward(delta: Delta): Unit = {
        synchronized {
          currentDelta = currentDelta |+| Some(delta)
        }
      }
    }

    type Output = Batch.Aux[SharedBatch#Data, SharedBatch#Delta]

    type SharedBatch <: ReferenceCount

    protected def cachedForward(input: Input): SharedBatch

    override def forward(input: Input): Output = {
      val sharedBatch = cache.get(input) match {
        case null =>
          val sharedBatch: SharedBatch = cachedForward(input)
          cache.put(input, sharedBatch)
          sharedBatch
        case sharedBatch =>
          sharedBatch.synchronized {
            sharedBatch.count += 1
          }
          sharedBatch
      }
      val checked = Option[sharedBatch.Output](sharedBatch.checked).getOrElse[sharedBatch.Output](sharedBatch)
      checked.asInstanceOf[Output with Nothing]
    }
  }

  import Batch._

  final case class Compose[A <: Batch, B <: Batch, C <: Batch](f: Differentiable.Aux[B, C],
                                                               g: Differentiable.Aux[A, B])
      extends Differentiable {
    override type Input = A
    override type Output = C

    override def forward(input: A): C = {
      f.forward(g.forward(input))
    }
  }

  final case class CConsHead[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
  TailDelta <: shapeless.Coproduct](
      ccons: Differentiable.Aux[Input0,
                                Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
  ) extends Differentiable {

    final class Output(upstream: Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
        extends Batch {
      override type Data = HeadData
      override type Delta = HeadDelta
      type Input >: Input0

      val value =
        upstream.value.asInstanceOf[shapeless.Inl[HeadData, TailData]].head

      override def backward(delta: Delta): Unit = {
        upstream.backward(shapeless.Inl(delta))
      }

      override def close(): Unit = {
        upstream.close()
      }

    }

    type Input = Input0

    override def forward(input: Input): Output = {
      new Output(ccons.forward(input))
    }

  }

  final case class CConsTail[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
  TailDelta <: shapeless.Coproduct](
      ccons: Differentiable.Aux[Input0,
                                Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
  ) extends Differentiable {

    final class Output(upstream: Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
        extends Batch {
      override type Data = TailData
      override type Delta = TailDelta
      type Input >: Input0

      val value =
        upstream.value.asInstanceOf[shapeless.Inr[TailData, TailData]].tail

      override def backward(delta: Delta): Unit = {
        upstream.backward(shapeless.Inr(delta))
      }

      override def close(): Unit = {
        upstream.close()
      }
    }

    type Input = Input0

    override def forward(input: Input): Output = {
      new Output(ccons.forward(input))
    }

  }

  final case class IsInl[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
  TailDelta <: shapeless.Coproduct](
      ccons: Differentiable.Aux[Input0,
                                Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
  ) extends Differentiable {

    final class Output(upstream: Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
        extends BooleanBatch {
      type Input >: Input0
      val value = upstream.value match {
        case shapeless.Inl(_) => Eval.now(true)
        case shapeless.Inr(_) => Eval.now(false)
      }

      override def backward(delta: Eval[scala.Boolean]): Unit = {}

      override def close(): Unit = {
        upstream.close()
      }
    }

    type Input = Input0

    override def forward(input: Input): Output = {
      new Output(ccons.forward(input))
    }
  }

  final case class DifferentiableInr[Input0 <: Batch, TailData <: shapeless.Coproduct,
  TailDelta <: shapeless.Coproduct](tail: Differentiable.Aux[Input0, Batch.Aux[TailData, TailDelta]])
      extends Differentiable {

    type Input = Input0

    final class Output(tailBatch: Batch.Aux[TailData, TailDelta]) extends Batch {
      def value = shapeless.Inr(tailBatch.value: TailData)

      type Data = shapeless.Inr[Nothing, TailData]
      type Delta = shapeless.:+:[scala.Any, TailDelta]

      override def backward(delta: shapeless.:+:[scala.Any, TailDelta]): Unit = {
        delta match {
          case shapeless.Inr(tailDelta) => tailBatch.backward(tailDelta)
          case shapeless.Inl(_) =>
        }
      }

      override def close(): Unit = {
        tailBatch.close()
      }
    }

    override def forward(input: Input0): Output = {
      new Output(tail.forward(input))
    }

  }

  final case class DifferentiableInl[Input0 <: Batch, HeadData, HeadDelta](
      head: Differentiable.Aux[Input0, Batch.Aux[HeadData, HeadDelta]])
      extends Differentiable {

    type Input = Input0

    final class Output(headBatch: Batch.Aux[HeadData, HeadDelta]) extends Batch {
      def value = shapeless.Inl(headBatch.value: HeadData)

      type Data = shapeless.Inl[HeadData, Nothing]
      type Delta = shapeless.:+:[HeadDelta, shapeless.Coproduct]

      override def backward(delta: shapeless.:+:[HeadDelta, Coproduct]): Unit = {
        delta match {
          case shapeless.Inl(headDelta) => headBatch.backward(headDelta)
          case shapeless.Inr(_) =>
        }
      }

      override def close(): Unit = {
        headBatch.close()
      }
    }

    override def forward(input: Input0): Output = {
      new Output(head.forward(input))
    }

  }

  final case class DifferentiableHNil[Input0 <: Batch]() extends Differentiable with Batch {
    override type Input = Input0

    override type Data = shapeless.HNil

    override type Delta = shapeless.CNil

    override type Output = Batch.Aux[Data, Delta]

    override def forward(input: Input): Output = this

    override def backward(delta: Delta): Unit = {}

    override def value = shapeless.HNil

    override def close(): Unit = {}
  }

  object DifferentiableHCons {

    final case class Head[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList,
    TailDelta <: shapeless.Coproduct](
        differentiableHCons: Differentiable.Aux[
          Input0,
          Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
    ) extends Differentiable {
      override type Input = Input0

      final class Output(upstream: Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
          extends Batch {
        override def backward(delta: Delta): Unit = {
          upstream.backward(shapeless.Inl(delta))
        }

        override def value: Data = {
          upstream.value.head
        }

        override type Data = HeadData
        override type Delta = HeadDelta

        override def close(): Unit = {
          upstream.close()
        }

      }

      override def forward(input: Input) = {
        new Output(differentiableHCons.forward(input))
      }
    }

    final case class Tail[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList,
    TailDelta <: shapeless.Coproduct](
        differentiableHCons: Differentiable.Aux[
          Input0,
          Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
    ) extends Differentiable {
      override type Input = Input0

      final class Output(upstream: Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
          extends Batch {
        override def backward(delta: Delta): Unit = {
          upstream.backward(shapeless.Inr(delta))
        }

        override def value: Data = {
          upstream.value.tail
        }

        override def close(): Unit = {
          upstream.close()
        }

        override type Data = TailData
        override type Delta = TailDelta
      }

      override def forward(input: Input) = {
        new Output(differentiableHCons.forward(input))
      }
    }

  }

  final case class DifferentiableHCons[Input0 <: Batch,
                                       HeadData,
                                       HeadDelta,
                                       TailData <: shapeless.HList,
                                       TailDelta <: shapeless.Coproduct](
      head: Differentiable.Aux[Input0, Batch.Aux[HeadData, HeadDelta]],
      tail: Differentiable.Aux[Input0, Batch.Aux[TailData, TailDelta]]
  ) extends Differentiable {
    override type Input = Input0

    final class Output(headBatch: Batch.Aux[HeadData, HeadDelta], tailBatch: Batch.Aux[TailData, TailDelta])
        extends Batch {
      override def backward(delta: Delta): Unit = {
        delta match {
          case shapeless.Inl(headDelta) =>
            headBatch.backward(headDelta)
          case shapeless.Inr(tailDelta) =>
            tailBatch.backward(tailDelta)
        }
      }

      override def value: Data = {
        headBatch.value :: tailBatch.value
      }

      override def close(): Unit = {
        headBatch.close()
        tailBatch.close()
      }

      override type Data = shapeless.::[HeadData, TailData]
      override type Delta = shapeless.:+:[HeadDelta, TailDelta]
    }

    override def forward(input: Input) = {
      new Output(head.forward(input), tail.forward(input))
    }

  }

  final case class Literal[Data0](value0: Data0) extends Differentiable with Batch {
    override type Data = Data0
    override type Delta = Any
    override type Input = Batch
    override type Output = this.type

    override def value: Data = value0

    override def forward(input: Input): Output = this

    override def backward(delta: Delta): Unit = {}

    override def close(): Unit = {}
  }

  final case class DoubleLessThanDouble[Input0 <: Batch](
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  ) extends Cached
      with Differentiable {

    final class SharedBatch(override val input: Input0,
                            upstream1: Batch.Aux[Eval[scala.Double], Eval[scala.Double]],
                            upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]])
        extends MonoidBatch
        with BooleanBatch {
      type Input >: Input0
      val value = upstream1.value.map2(upstream2.value)(_ < _).memoize

      override protected def cachedClose(): Unit = {
        upstream1.close()
        upstream2.close()
      }

      override protected def cachedBackward(delta: Eval[scala.Boolean]): Unit = {
        upstream1.backward(Eval.now(0.0))
        upstream2.backward(Eval.now(0.0))
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      new SharedBatch(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }
  }

  final case class DoubleAddDouble[Input0 <: Batch](
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  ) extends Cached
      with Differentiable {

    final class SharedBatch(override val input: Input0,
                            upstream1: Batch.Aux[Eval[scala.Double], Eval[scala.Double]],
                            upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]])
        extends MonoidBatch
        with DoubleBatch {
      type Input >: Input0
      val value = upstream1.value.map2(upstream2.value)(_ + _)

      override protected def cachedClose(): Unit = {
        upstream1.close()
        upstream2.close()
      }

      override protected def cachedBackward(delta: Eval[scala.Double]): Unit = {
        upstream1.backward(delta)
        upstream2.backward(delta)
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      new SharedBatch(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }

  }

  final case class DoubleSubtractDouble[Input0 <: Batch](
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  ) extends Cached
      with Differentiable {

    final class SharedBatch(override val input: Input0,
                            upstream1: Batch.Aux[Eval[scala.Double], Eval[scala.Double]],
                            upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]])
        extends MonoidBatch
        with DoubleBatch {
      type Input >: Input0
      val value = upstream1.value.map2(upstream2.value)(_ - _)

      override protected def cachedClose(): Unit = {
        upstream1.close()
        upstream2.close()
      }

      override protected def cachedBackward(delta: Eval[scala.Double]): Unit = {
        upstream1.backward(delta)
        upstream2.backward(delta.map(-_))
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      new SharedBatch(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }

  }

  final case class DoubleMultiplyDouble[Input0 <: Batch](
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  ) extends Cached
      with Differentiable {

    final class SharedBatch(override val input: Input0,
                            upstream1: Batch.Aux[Eval[scala.Double], Eval[scala.Double]],
                            upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]])
        extends MonoidBatch
        with DoubleBatch {
      type Input >: Input0
      val value = upstream1.value.map2(upstream2.value)(_ * _)

      override protected def cachedClose(): Unit = {
        upstream1.close()
        upstream2.close()
      }

      override protected def cachedBackward(delta: Eval[scala.Double]): Unit = {
        val a = upstream1.value
        val b = upstream2.value
        upstream1.backward(delta.map2(b)(_ * _))
        upstream2.backward(delta.map2(a)(_ * _))
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      new SharedBatch(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }

  }

  final case class DoubleReciprocal[Input0 <: Batch](
      differentiableDouble: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]])
      extends Cached
      with Differentiable {

    final class SharedBatch(override val input: Input0, upstream: Batch.Aux[Eval[scala.Double], Eval[scala.Double]])
        extends MonoidBatch
        with DoubleBatch {
      type Input >: Input0
      val value = upstream.value.map(1.0 / _)

      override protected def cachedClose(): Unit = {
        upstream.close()
      }

      override protected def cachedBackward(delta: Eval[scala.Double]): Unit = {
        val a = upstream.value
        upstream.backward(delta.map2(a) { (outputDeltaValue: scala.Double, aValue: scala.Double) =>
          -outputDeltaValue / (aValue * aValue)
        })
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      val upstream = differentiableDouble.forward(input)
      new SharedBatch(input, upstream)
    }
  }

  final case class DoubleNegative[Input0 <: Batch](
      differentiableDouble: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]])
      extends Cached
      with Differentiable {

    final class SharedBatch(override val input: Input0, upstream: Batch.Aux[Eval[scala.Double], Eval[scala.Double]])
        extends MonoidBatch
        with DoubleBatch {
      type Input >: Input0
      val value = upstream.value.map(-_)

      override protected def cachedClose(): Unit = {
        upstream.close()
      }

      override protected def cachedBackward(delta: Eval[scala.Double]): Unit = {
        upstream.backward(delta.map(-_))
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      val upstream = differentiableDouble.forward(input)
      new SharedBatch(input, upstream)
    }
  }

  final case class DoubleWeight[Input0 <: Batch](var rawValue: scala.Double)(implicit learningRate: LearningRate)
      extends Differentiable
      with DoubleBatch {
    override type Input = Input0
    override type Output = this.type

    override def forward(any: Input) = this

    override def backward(delta: Delta): Unit = {
      rawValue -= delta.value * learningRate()
    }

    override def value = Eval.now(rawValue)

    override def close(): Unit = {}

  }

  final case class BooleanWeight[Input0 <: Batch](var rawValue: scala.Boolean)
      extends Differentiable
      with BooleanBatch {
    override type Input = Input0
    override type Output = BooleanWeight[Input0]

    override def forward(any: Input) = this

    override def backward(delta: Delta): Unit = {
      rawValue ^= delta.value
    }

    override def value = Eval.now(rawValue)

    override def close(): Unit = {}

  }

  final case class Array2DWeight[Input0 <: Batch](var rawValue: INDArray)(implicit learningRate: LearningRate)
      extends Differentiable
      with Array2DBatch {
    override type Input = Input0
    override type Output = Array2DWeight[Input0]

    override def value = Eval.now(rawValue)

    override def forward(any: Input) = this

    override def backward(delta: Delta): Unit = {
      rawValue -= delta.value * learningRate()
    }

    override def close(): Unit = {}

  }

  object Array2DWeight {
    def randn[Input <: Batch](numberOfRows: Int, numberOfColumns: Int)(
        implicit learningRate: LearningRate): Array2DWeight[Input] = {
      new Array2DWeight[Input](Nd4j.randn(numberOfRows, numberOfColumns))
    }

    def zeros[Input <: Batch](numberOfRows: Int, numberOfColumns: Int)(
        implicit learningRate: LearningRate): Array2DWeight[Input] = {
      new Array2DWeight[Input](Nd4j.zeros(numberOfRows, numberOfColumns))
    }

    def apply[Input <: Batch](nativeArray: Array[Array[scala.Double]])(
        implicit learningRate: LearningRate): Array2DWeight[Input] = {
      new Array2DWeight[Input](nativeArray.toNDArray)
    }

  }

  final case class Dot[Input0 <: Batch](
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]]
  ) extends Differentiable
      with Cached {

    final class SharedBatch(override val input: Input0,
                            upstream1: Batch.Aux[Eval[INDArray], Eval[INDArray]],
                            upstream2: Batch.Aux[Eval[INDArray], Eval[INDArray]])
        extends Array2DBatch
        with SemigroupBatch {
      override val value = upstream1.value.map2(upstream2.value)(_ dot _).memoize

      type Input >: Input0

      override protected def cachedClose(): Unit = {
        upstream1.close()
        upstream2.close()
      }

      override protected def cachedBackward(outputDelta: Eval[INDArray]): Unit = {
        val b = upstream2.value
        upstream1.backward(
          outputDelta
            .map2(b) {
              _ dot _.T
            }
            .memoize)
        val a = upstream1.value
        upstream2.backward(
          outputDelta
            .flatMap[INDArray] { outputDeltaValue =>
              a.map { aData =>
                aData.T.dot(outputDeltaValue)
              }
            }
            .memoize)
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      new SharedBatch(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }
  }

  private def sumAs(outputDeltaValue: INDArray, shape: Array[Int]) =
    shape match {
      case Array(1, 1) => outputDeltaValue.sum(0, 1)
      case Array(_, 1) => outputDeltaValue.sum(1)
      case Array(1, _) => outputDeltaValue.sum(0)
      case Array(_, _) => outputDeltaValue
    }

  final case class Array2DMultiplyArray2D[Input0 <: Batch](
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]]
  ) extends Differentiable
      with Cached {

    final class SharedBatch(override val input: Input0,
                            upstream1: Batch.Aux[Eval[INDArray], Eval[INDArray]],
                            upstream2: Batch.Aux[Eval[INDArray], Eval[INDArray]])
        extends Array2DBatch
        with SemigroupBatch {
      val value = {
        upstream1.value
          .map2(upstream2.value) { (aValue, bValue) =>
            val Array(aRows, aColumns) = aValue.shape()
            val Array(bRows, bColumns) = bValue.shape()
            val newShape =
              Array(math.max(aRows, bRows), math.max(aColumns, bColumns))
            aValue.broadcast(newShape: _*) * bValue.broadcast(newShape: _*)
          }
          .memoize
      }

      type Input >: Input0

      override protected def cachedClose(): Unit = {
        upstream1.close()
        upstream2.close()
      }

      override protected def cachedBackward(outputDelta: Eval[INDArray]): Unit = {
        val a = upstream1.value
        val b = upstream2.value
        upstream1.backward(
          Applicative[Eval]
            .map3(outputDelta, a, b) { (outputDeltaValue, aData, bData) =>
              sumAs(bData.broadcast(outputDeltaValue.shape(): _*) * outputDeltaValue, aData.shape())
            }
            .memoize)
        upstream2.backward(
          Applicative[Eval]
            .map3(outputDelta, a, b) { (outputDeltaValue, aData, bData) =>
              sumAs(aData.broadcast(outputDeltaValue.shape(): _*) * outputDeltaValue, bData.shape())
            }
            .memoize)
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      new SharedBatch(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }
  }

  final case class Array2DAddArray2D[Input0 <: Batch](
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]]
  ) extends Differentiable
      with Cached {

    final class SharedBatch(override val input: Input0,
                            upstream1: Batch.Aux[Eval[INDArray], Eval[INDArray]],
                            upstream2: Batch.Aux[Eval[INDArray], Eval[INDArray]])
        extends Array2DBatch
        with SemigroupBatch {
      val value = {
        Applicative[Eval]
          .map2(upstream1.value, upstream2.value) { (aValue, bValue) =>
            val Array(aRows, aColumns) = aValue.shape()
            val Array(bRows, bColumns) = bValue.shape()
            val newShape =
              Array(math.max(aRows, bRows), math.max(aColumns, bColumns))
            aValue.broadcast(newShape: _*) + bValue.broadcast(newShape: _*)
          }
          .memoize
      }

      type Input >: Input0

      override protected def cachedClose(): Unit = {
        upstream1.close()
        upstream2.close()
      }

      override protected def cachedBackward(outputDelta: Eval[INDArray]): Unit = {
        val sumAsOriginalShape = { (outputDeltaValue: INDArray, upstreamValue: INDArray) =>
          sumAs(outputDeltaValue, upstreamValue.shape)
        }
        upstream1.backward(outputDelta.map2(upstream1.value)(sumAsOriginalShape))
        upstream2.backward(outputDelta.map2(upstream2.value)(sumAsOriginalShape))
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      new SharedBatch(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }
  }

  final case class Array2DMultiplyDouble[Input0 <: Batch](
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  ) extends Differentiable
      with Cached {

    final class SharedBatch(override val input: Input0,
                            upstream1: Batch.Aux[Eval[INDArray], Eval[INDArray]],
                            upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]])
        extends Array2DBatch
        with SemigroupBatch {
      val value = upstream1.value.map2(upstream2.value)(_ * _).memoize

      type Input >: Input0

      override protected def cachedClose(): Unit = {
        upstream1.close()
        upstream2.close()
      }

      override protected def cachedBackward(outputDelta: Eval[INDArray]): Unit = {
        val a = upstream1.value
        val b = upstream2.value

        val aDelta = outputDelta.map2(b)(_ * _).memoize
        upstream1.backward(aDelta)
        val bDelta = outputDelta
          .map2(a) { (outputDeltaValue, aValue) =>
            (aValue * outputDeltaValue).sumT
          }
          .memoize
        upstream2.backward(bDelta)
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      new SharedBatch(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }
  }

  final case class Array2DMaxDouble[Input0 <: Batch](
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  ) extends Differentiable
      with Cached {

    final class SharedBatch(override val input: Input0,
                            upstream1: Batch.Aux[Eval[INDArray], Eval[INDArray]],
                            upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]])
        extends Array2DBatch
        with SemigroupBatch {
      val value = upstream1.value.map2(upstream2.value)(Transforms.max).memoize

      type Input >: Input0

      override protected def cachedClose(): Unit = {
        upstream1.close()
        upstream2.close()
      }

      override protected def cachedBackward(outputDelta: Eval[INDArray]): Unit = {
        val a = upstream1.value
        val b = upstream2.value
        upstream1.backward(
          Applicative[Eval].map3(outputDelta, a, b) { (outputDeltaValue, aValue, bValue) =>
            (aValue gt bValue) * outputDeltaValue
          }
        )
        upstream2.backward(
          Applicative[Eval].map3(outputDelta, a, b) { (outputDeltaValue, aValue, bValue) =>
            ((aValue lt bValue) * outputDeltaValue).sumT
          }
        )
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      new SharedBatch(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }
  }

  final case class Array2DAddDouble[Input0 <: Batch](
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  ) extends Differentiable
      with Cached {

    final class SharedBatch(override val input: Input0,
                            upstream1: Batch.Aux[Eval[INDArray], Eval[INDArray]],
                            upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]])
        extends Array2DBatch
        with SemigroupBatch {
      val value = upstream1.value.map2(upstream2.value)(_ + _).memoize

      type Input >: Input0

      override protected def cachedClose(): Unit = {
        upstream1.close()
        upstream2.close()
      }

      override protected def cachedBackward(outputDelta: Eval[INDArray]): Unit = {
        upstream1.backward(outputDelta)
        upstream2.backward(outputDelta.map(_.sumT))
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      new SharedBatch(input, leftHandSide.forward(input), rightHandSide.forward(input))
    }
  }

  final case class Array2DReciprocal[Input0 <: Batch](
      differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]])
      extends Cached
      with Differentiable {

    final class SharedBatch(override val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[INDArray]])
        extends Array2DBatch
        with SemigroupBatch {
      val value = upstream.value.map(_ rdiv 1.0).memoize

      type Input >: Input0

      override protected def cachedClose(): Unit = {
        upstream.close()
      }

      override protected def cachedBackward(outputDelta: Eval[INDArray]): Unit = {
        val upstreamValue = upstream.value
        upstream.backward(
          outputDelta
            .map2(upstream.value) { (outputDeltaValue, aValue) =>
              -outputDeltaValue / (aValue * aValue)
            }
            .memoize
        )
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      val upstream = differentiableArray2D.forward(input)
      new SharedBatch(input, upstream)
    }
  }

  final case class ReduceSum[Input0 <: Batch](
      differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]])
      extends Cached
      with Differentiable {

    final class SharedBatch(override val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[INDArray]])
        extends MonoidBatch
        with DoubleBatch {
      type Input >: Input0
      val value = upstream.value.map(_.sumT).memoize

      override protected def cachedClose(): Unit = {
        upstream.close()
      }

      override protected def cachedBackward(outputDelta: Eval[scala.Double]): Unit = {
        upstream.backward(
          outputDelta
            .map2(upstream.value) { (outputDeltaValue, aValue) =>
              Nd4j.valueArrayOf(aValue.shape(), outputDeltaValue)
            }
            .memoize)
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      val upstream = differentiableArray2D.forward(input)
      new SharedBatch(input, upstream)
    }
  }

  final case class Sum[Input0 <: Batch](
      differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]],
      dimensions: Seq[Int])
      extends Cached
      with Differentiable {

    final class SharedBatch(override val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[INDArray]])
        extends Array2DBatch
        with SemigroupBatch {
      val value = upstream.value.map(_.sum(dimensions: _*)).memoize

      type Input >: Input0

      override protected def cachedClose(): Unit = {
        upstream.close()
      }

      override protected def cachedBackward(outputDelta: Eval[INDArray]): Unit = {
        val a = upstream.value
        upstream.backward(
          outputDelta
            .map2(a) { (outputDeltaValue, aValue) =>
              outputDeltaValue.broadcast(aValue.shape: _*)
            }
            .memoize)
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      val upstream = differentiableArray2D.forward(input)
      new SharedBatch(input, upstream)
    }
  }

  final case class Array2DNegative[Input0 <: Batch](
      differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]])
      extends Cached
      with Differentiable {

    final class SharedBatch(override val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[INDArray]])
        extends Array2DBatch
        with SemigroupBatch {
      val value = upstream.value.map(-_).memoize

      type Input >: Input0

      override protected def cachedClose(): Unit = {
        upstream.close()
      }

      override protected def cachedBackward(outputDelta: Eval[INDArray]): Unit = {
        upstream.backward(outputDelta.map(-_).memoize)
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      val upstream = differentiableArray2D.forward(input)
      new SharedBatch(input, upstream)
    }
  }

  final case class Array2DLog[Input0 <: Batch](
      differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]])
      extends Cached
      with Differentiable {

    final class SharedBatch(override val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[INDArray]])
        extends Array2DBatch
        with SemigroupBatch {
      val value = upstream.value.map(Transforms.log).memoize

      type Input >: Input0

      override protected def cachedClose(): Unit = {
        upstream.close()
      }

      override protected def cachedBackward(outputDelta: Eval[INDArray]): Unit = {
        upstream.backward(outputDelta.map2(upstream.value)(_ / _).memoize)
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      val upstream = differentiableArray2D.forward(input)
      new SharedBatch(input, upstream)
    }
  }

  final case class Array2DExp[Input0 <: Batch](
      differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[INDArray]]])
      extends Cached
      with Differentiable {

    final class SharedBatch(override val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[INDArray]])
        extends Array2DBatch
        with SemigroupBatch {
      val value = upstream.value.map(Transforms.exp).memoize

      type Input >: Input0

      override protected def cachedClose(): Unit = {
        upstream.close()
      }

      override protected def cachedBackward(outputDelta: Eval[INDArray]): Unit = {
        upstream.backward(value.map2(outputDelta)(_ * _).memoize)
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      val upstream = differentiableArray2D.forward(input)
      new SharedBatch(input, upstream)
    }
  }

  final case class If[Input0 <: Batch, Output0 <: Batch](
      condition: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]],
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

  final case class Not[Input0 <: Batch](
      differentiableBoolean: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]])
      extends Cached
      with Differentiable {

    final class SharedBatch(override val input: Input0, upstream: Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]])
        extends MonoidBatch
        with BooleanBatch {
      type Input >: Input0
      val value = upstream.value.map(!_)

      override protected def cachedBackward(delta: Eval[scala.Boolean]): Unit = {
        upstream.backward(delta.map(!_))
      }

      override protected def cachedClose(): Unit = {
        upstream.close()
      }
    }

    type Input = Input0

    override protected def cachedForward(input: Input): SharedBatch = {
      val upstream = differentiableBoolean.forward(input)
      new SharedBatch(input, upstream)
    }
  }

  trait LearningRate {
    def apply(): scala.Double
  }

}

trait Differentiable {

  import Differentiable._

  type Input <: Batch

  type Output <: Batch

  def forward(input: Input): Output

}
