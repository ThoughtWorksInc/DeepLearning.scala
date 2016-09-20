package com.thoughtworks.deepLearning

import cats.{Applicative, Eval, Monoid}

import scala.language.existentials
import scala.language.implicitConversions
import scala.language.higherKinds
import cats.implicits._
import com.thoughtworks.deepLearning.Differentiable.Batch.Aux
import com.thoughtworks.deepLearning.Dsl.DslFunction
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import shapeless.{::, DepFn0, Generic, HNil}
import simulacrum.typeclass


object Differentiable {

  //
  //  @typeclass
  //  trait FromDsl[F[_ <: Dsl] <: DslFunction] extends DepFn0
  //
  //  object FromDsl {
  //
  //    type Aux[F[_ <: Dsl] <: DslFunction, Out0] = FromDsl[F] {
  //      type Out = Out0
  //    }
  //
  //    implicit def doubleFromDslFactory[F[_ <: Dsl] <: DslFunction]
  //    (
  //      implicit constraint: F[SymbolicDsl.Aux[Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]] <:< DslFunction.Aux[SymbolicDsl.Aux[Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]#Double, F[SymbolicDsl.Aux[Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]]#Out],
  //      learningRate: LearningRate,
  //      g: Generic.Aux[F[SymbolicDsl.Aux[Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]], SymbolicDsl.Aux[Batch.Aux[Eval[scala.Double], Eval[scala.Double]]] :: HNil]
  //    ): FromDsl.Aux[F, F[SymbolicDsl.Aux[Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]]#Out] = {
  //      new FromDsl[F] {
  //        type Out = F[SymbolicDsl.Aux[Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]]#Out
  //
  //        override def apply() = {
  //          val dsl = SymbolicDsl[Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  //          g.from(dsl :: HNil)(dsl.Double.specialize(Id[Eval[scala.Double], Eval[scala.Double]]))
  //        }
  //      }
  //    }
  //
  //    implicit def array2DFromDslFactory[F[_ <: Dsl] <: DslFunction]
  //    (
  //      implicit constraint: F[SymbolicDsl.Aux[Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]] <:< DslFunction.Aux[SymbolicDsl.Aux[Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]#Array2D, F[SymbolicDsl.Aux[Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]]#Out],
  //      learningRate: LearningRate,
  //      g: Generic.Aux[F[SymbolicDsl.Aux[Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]], SymbolicDsl.Aux[Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]] :: HNil]
  //    ): FromDsl.Aux[F, F[SymbolicDsl.Aux[Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]]#Out] = {
  //      new FromDsl[F] {
  //        type Out = F[SymbolicDsl.Aux[Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]]#Out
  //
  //        override def apply() = {
  //          val dsl = SymbolicDsl[Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]
  //          g.from(dsl :: HNil)(dsl.Array2D.specialize(Id[Eval[INDArray], Eval[Option[INDArray]]]))
  //        }
  //      }
  //    }
  //
  //  }
  //
  //  def fromDsl[F[_ <: Dsl] <: DslFunction](implicit fromDsl0: FromDsl[F]): fromDsl0.Out = {
  //    fromDsl0()
  //  }

  object Batch {
    type Aux[+Data0, -Delta0] = Batch {
      type Data <: Data0
      type Delta >: Delta0
    }

    case object HNilBatch extends Batch {
      override type Data = shapeless.HNil
      override type Delta = shapeless.HNil

      override def backward(delta: Delta): Unit = ???

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

      final def monoid: Monoid[Delta] = ???

    }

    trait DoubleBatch extends Batch {

      override type Data = Eval[scala.Double]

      override type Delta = Eval[scala.Double]

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

  final case class DifferentiableHCons[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList, TailDelta <: shapeless.HList]
  (
    head: Differentiable.Aux[Input0, Batch.Aux[HeadData, HeadDelta]],
    tail: Differentiable.Aux[Input0, Batch.Aux[TailData, TailDelta]]
  ) extends Differentiable {
    override type Input = Input0

    final case class Output(headBatch: Batch.Aux[HeadData, HeadDelta], tailBatch: Batch.Aux[TailData, TailDelta]) extends Batch {
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
      Output(head.forward(input), tail.forward(input))
    }

  }


  final case class DifferentiableHNil[Input0 <: Batch]() extends Differentiable {
    override type Input = Input0
    override type Output = Batch.Aux[HNil, HNil]

    override def forward(input: Input): Output = HNilBatch
  }

  //
  //  object DifferentiableDouble {
  //    type Aux[Input0 <: Batch] = DifferentiableDouble {
  //      type Input = Input0
  //    }
  //
  //    final case class LessThan[Input0 <: Batch]
  //    (
  //      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
  //      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  //    ) extends Cached with DifferentiableBoolean {
  //
  //      final class Output(val input: Input0, upstream1: Batch.Aux[Eval[scala.Double], Eval[scala.Double]], upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with BooleanBatch {
  //        type Input >: Input0
  //        val value = upstream1.value.map2(upstream2.value)(_ < _).memoize
  //
  //        override protected def cachedBackward(delta: Eval[scala.Boolean]): Unit = {
  //          upstream1.backward(Eval.now(0.0))
  //          upstream2.backward(Eval.now(0.0))
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
  //      }
  //    }
  //
  //    final case class Add[Input0 <: Batch]
  //    (
  //      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
  //      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  //    ) extends Cached with DifferentiableDouble {
  //
  //      final class Output(val input: Input0, upstream1: Batch.Aux[Eval[scala.Double], Eval[scala.Double]], upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with DoubleBatch {
  //        type Input >: Input0
  //        val value = upstream1.value.map2(upstream2.value)(_ + _)
  //
  //        override protected def cachedBackward(delta: Eval[scala.Double]): Unit = {
  //          upstream1.backward(delta)
  //          upstream2.backward(delta)
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
  //      }
  //
  //    }
  //
  //    final case class Subtract[Input0 <: Batch]
  //    (
  //      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
  //      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  //    ) extends Cached with DifferentiableDouble {
  //
  //      final class Output(val input: Input0, upstream1: Batch.Aux[Eval[scala.Double], Eval[scala.Double]], upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with DoubleBatch {
  //        type Input >: Input0
  //        val value = upstream1.value.map2(upstream2.value)(_ - _)
  //
  //        override protected def cachedBackward(delta: Eval[scala.Double]): Unit = {
  //          upstream1.backward(delta)
  //          upstream2.backward(delta.map(-_))
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
  //      }
  //
  //    }
  //
  //    final case class Multiply[Input0 <: Batch]
  //    (
  //      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
  //      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  //    ) extends Cached with DifferentiableDouble {
  //
  //      final class Output(val input: Input0, upstream1: Batch.Aux[Eval[scala.Double], Eval[scala.Double]], upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with DoubleBatch {
  //        type Input >: Input0
  //        val value = upstream1.value.map2(upstream2.value)(_ * _)
  //
  //        override protected def cachedBackward(delta: Eval[scala.Double]): Unit = {
  //          val a = upstream1.value
  //          val b = upstream2.value
  //          upstream1.backward(delta.map2(b)(_ * _))
  //          upstream2.backward(delta.map2(a)(_ * _))
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
  //      }
  //
  //    }
  //
  //    final case class Reciprocal[Input0 <: Batch](differentiableDouble: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]) extends Cached with DifferentiableDouble {
  //
  //      final class Output(val input: Input0, upstream: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with DoubleBatch {
  //        type Input >: Input0
  //        val value = upstream.value.map(1.0 / _)
  //
  //        override protected def cachedBackward(delta: Eval[scala.Double]): Unit = {
  //          val a = upstream.value
  //          upstream.backward(delta.map2(a) {
  //            (outputDeltaValue: scala.Double, aValue: scala.Double) =>
  //              -outputDeltaValue / (aValue * aValue)
  //          })
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        val upstream = differentiableDouble.forward(input)
  //        new Output(input, upstream)
  //      }
  //    }
  //
  //    final case class Negative[Input0 <: Batch](differentiableDouble: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]) extends Cached with DifferentiableDouble {
  //
  //      final class Output(val input: Input0, upstream: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with DoubleBatch {
  //        type Input >: Input0
  //        val value = upstream.value.map(-_)
  //
  //        override protected def cachedBackward(delta: Eval[scala.Double]): Unit = {
  //          upstream.backward(delta.map(-_))
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        val upstream = differentiableDouble.forward(input)
  //        new Output(input, upstream)
  //      }
  //    }
  //
  //    final case class DoubleLiteral[Input0 <: Batch](rawValue: scala.Double) extends DifferentiableDouble with DoubleBatch {
  //      override type Input = Input0
  //      override type Output = DoubleLiteral[Input0]
  //
  //      override def forward(any: Input) = this
  //
  //      override def backward(delta: Delta): Unit = {}
  //
  //      override def value = Eval.now(rawValue)
  //    }
  //
  //    final case class DoubleWeight[Input0 <: Batch](var rawValue: scala.Double)(implicit learningRate: LearningRate) extends DifferentiableDouble with DoubleBatch {
  //      override type Input = Input0
  //      override type Output = DoubleWeight[Input0]
  //
  //      override def forward(any: Input) = this
  //
  //      override def backward(delta: Delta): Unit = {
  //        rawValue -= delta.value * learningRate()
  //      }
  //
  //      override def value = Eval.now(rawValue)
  //
  //    }
  //
  //    final case class DoubleOps[Input0 <: Batch](generic: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]) extends DifferentiableDouble {
  //      override type Output = generic.Output
  //      override type Input = Input0
  //
  //      override def forward(input: Input0): generic.Output = {
  //        generic.forward(input)
  //      }
  //    }
  //
  //  }
  //
  //  trait DifferentiableDouble extends Differentiable {
  //
  //    import DifferentiableDouble._
  //
  //    override type Output <: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]
  //
  //    override def unary_- = {
  //      Negative[Input](this)
  //    }
  //
  //    override def -(rightHandSide: Double) = {
  //      Subtract[Input](this, rightHandSide)
  //    }
  //
  //    override def <(rightHandSide: Double) = {
  //      LessThan[Input](this, rightHandSide)
  //    }
  //
  //    override def +(rightHandSide: Double) = {
  //      Add[Input](this, rightHandSide)
  //    }
  //
  //    override def /(rightHandSide: Double) = {
  //      Multiply[Input](this, Reciprocal[Input](rightHandSide))
  //    }
  //
  //    override def /(rightHandSide: Array2D) = {
  //      DifferentiableArray2D.MultiplyDouble[Input](DifferentiableArray2D.Reciprocal[Input](rightHandSide), this)
  //    }
  //
  //    override def *(rightHandSide: Double) = {
  //      Multiply[Input](this, rightHandSide)
  //    }
  //  }
  //
  //  object DifferentiableArray2D {
  //    type Aux[Input0 <: Batch] = DifferentiableArray2D {
  //      type Input = Input0
  //    }
  //
  //    final case class Array2DLiteral[Input0 <: Batch](rawValue: INDArray) extends DifferentiableArray2D with Array2DBatch {
  //      override type Input = Input0
  //      override type Output = Array2DLiteral[Input0]
  //
  //      override def value = Eval.now(rawValue)
  //
  //      override def forward(any: Input) = this
  //
  //      override def backward(delta: Delta): Unit = {}
  //    }
  //
  //    object Array2DLiteral {
  //      def apply[Input <: Batch](nativeArray: Array[Array[scala.Double]]): Array2DLiteral[Input] = {
  //        new Array2DLiteral[Input](nativeArray.toNDArray)
  //      }
  //    }
  //
  //    final case class Array2DWeight[Input0 <: Batch](var rawValue: INDArray)(implicit learningRate: LearningRate) extends DifferentiableArray2D with Array2DBatch {
  //      override type Input = Input0
  //      override type Output = Array2DWeight[Input0]
  //
  //      override def value = Eval.now(rawValue)
  //
  //      override def forward(any: Input) = this
  //
  //      override def backward(delta: Delta): Unit = {
  //        delta.value match {
  //          case Some(deltaValue) =>
  //            rawValue -= deltaValue * learningRate()
  //          case None =>
  //        }
  //      }
  //    }
  //
  //    object Array2DWeight {
  //      def randn[Input <: Batch](numberOfRows: Int, numberOfColumns: Int)(implicit learningRate: LearningRate): Array2DWeight[Input] = {
  //        new Array2DWeight[Input](Nd4j.randn(numberOfRows, numberOfColumns))
  //      }
  //
  //      def zeros[Input <: Batch](numberOfRows: Int, numberOfColumns: Int)(implicit learningRate: LearningRate): Array2DWeight[Input] = {
  //        new Array2DWeight[Input](Nd4j.zeros(numberOfRows, numberOfColumns))
  //      }
  //
  //      def apply[Input <: Batch](nativeArray: Array[Array[scala.Double]])(implicit learningRate: LearningRate): Array2DWeight[Input] = {
  //        new Array2DWeight[Input](nativeArray.toNDArray)
  //      }
  //
  //    }
  //
  //    final case class Array2DOps[Input0 <: Batch](generic: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends DifferentiableArray2D {
  //      override type Output = generic.Output
  //      override type Input = Input0
  //
  //      override def forward(input: Input0): generic.Output = {
  //        generic.forward(input)
  //      }
  //    }
  //
  //    final case class Dot[Input0 <: Batch]
  //    (
  //      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
  //      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]
  //    ) extends DifferentiableArray2D with Cached {
  //
  //      final class Output(val input: Input0, upstream1: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]], upstream2: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with Array2DBatch {
  //        type Input >: Input0
  //        val value = upstream1.value.map2(upstream2.value)(_ dot _).memoize
  //
  //        override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
  //          val b = upstream2.value
  //          upstream1.backward(outputDelta.flatMap[Option[INDArray]] {
  //            case None => Eval.now(None)
  //            case Some(outputDeltaValue) =>
  //              b.map {
  //                bData =>
  //                  Some(outputDeltaValue.dot(bData.T))
  //              }
  //          }.memoize)
  //          val a = upstream1.value
  //          upstream2.backward(outputDelta.flatMap[Option[INDArray]] {
  //            case None => Eval.now(None)
  //            case Some(outputDeltaValue) =>
  //              a.map {
  //                aData =>
  //                  Some(aData.T.dot(outputDeltaValue))
  //              }
  //          }.memoize)
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
  //      }
  //    }
  //
  //    private def sumAs(outputDeltaValue: INDArray, shape: Array[Int]) = shape match {
  //      case Array(1, 1) => outputDeltaValue.sum(0, 1)
  //      case Array(_, 1) => outputDeltaValue.sum(1)
  //      case Array(1, _) => outputDeltaValue.sum(0)
  //      case Array(_, _) => outputDeltaValue
  //    }
  //
  //    final case class MultiplyArray2D[Input0 <: Batch]
  //    (
  //      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
  //      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]
  //    ) extends DifferentiableArray2D with Cached {
  //
  //      final class Output(val input: Input0, upstream1: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]], upstream2: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with Array2DBatch {
  //        type Input >: Input0
  //        val value = upstream1.value.map2(upstream2.value)(_ * _).memoize
  //
  //        override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
  //          val a = upstream1.value
  //          val b = upstream2.value
  //          upstream1.backward(outputDelta.flatMap[Option[INDArray]] {
  //            case None => Eval.now(None)
  //            case Some(outputDeltaValue) =>
  //              a.map2(b) { (aData, bData) =>
  //                Some(sumAs(bData.broadcast(outputDeltaValue.shape(): _*) * outputDeltaValue, aData.shape()))
  //              }
  //          }.memoize)
  //          upstream2.backward(outputDelta.flatMap[Option[INDArray]] {
  //            case None => Eval.now(None)
  //            case Some(outputDeltaValue) =>
  //              a.map2(b) { (aData, bData) =>
  //                Some(sumAs(aData.broadcast(outputDeltaValue.shape(): _*) * outputDeltaValue, bData.shape()))
  //              }
  //          }.memoize)
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
  //      }
  //    }
  //
  //    final case class AddArray2D[Input0 <: Batch]
  //    (
  //      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
  //      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]
  //    ) extends DifferentiableArray2D with Cached {
  //
  //      final class Output(val input: Input0, upstream1: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]], upstream2: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with Array2DBatch {
  //        type Input >: Input0
  //        val value = upstream1.value.map2(upstream2.value)(_ + _).memoize
  //
  //        override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
  //          val sumAsOriginalShape = { (outputDeltaOption: Option[INDArray], aValue: INDArray) =>
  //            outputDeltaOption.map(sumAs(_, aValue.shape()))
  //          }
  //          upstream1.backward(outputDelta.map2(upstream1.value)(sumAsOriginalShape))
  //          upstream2.backward(outputDelta.map2(upstream2.value)(sumAsOriginalShape))
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
  //      }
  //    }
  //
  //    final case class MultiplyDouble[Input0 <: Batch]
  //    (
  //      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
  //      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  //    ) extends DifferentiableArray2D with Cached {
  //
  //      final class Output(val input: Input0, upstream1: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]], upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with Array2DBatch {
  //        type Input >: Input0
  //        val value = upstream1.value.map2(upstream2.value)(_ * _).memoize
  //
  //        override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
  //
  //          val a = upstream1.value
  //          val b = upstream2.value
  //
  //          val aDelta = outputDelta.flatMap[Option[INDArray]] {
  //            case None => Eval.now(None)
  //            case Some(outputDeltaValue) =>
  //              b.map {
  //                bData: scala.Double =>
  //                  Some(outputDeltaValue * bData)
  //              }
  //          }.memoize
  //          upstream1.backward(aDelta)
  //          val bDelta = outputDelta.flatMap[scala.Double] {
  //            case None => Eval.now(0.0)
  //            case Some(outputDeltaValue) =>
  //              a.map {
  //                aData: INDArray =>
  //                  (aData * outputDeltaValue).sumT
  //              }
  //          }.memoize
  //          upstream2.backward(bDelta)
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
  //      }
  //    }
  //
  //    final case class MaxDouble[Input0 <: Batch]
  //    (
  //      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
  //      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  //    ) extends DifferentiableArray2D with Cached {
  //
  //      final class Output(val input: Input0, upstream1: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]], upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with Array2DBatch {
  //        type Input >: Input0
  //        val value = upstream1.value.map2(upstream2.value)(Transforms.max).memoize
  //
  //        override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
  //          val a = upstream1.value
  //          val b = upstream2.value
  //          upstream1.backward(outputDelta.flatMap[Option[INDArray]] {
  //            case None => Eval.now(None)
  //            case Some(outputDeltaValue) =>
  //              Applicative[Eval].map2(a, b) {
  //                (aData: INDArray, bData: scala.Double) =>
  //                  Some((aData gt bData) * outputDeltaValue)
  //              }
  //          })
  //          upstream2.backward(outputDelta.flatMap[scala.Double] {
  //            case None => Eval.now(0)
  //            case Some(outputDeltaValue) =>
  //              Applicative[Eval].map2(a, b) {
  //                (aData: INDArray, bData: scala.Double) =>
  //                  ((aData lt bData) * outputDeltaValue).sumT
  //              }
  //          })
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
  //      }
  //    }
  //
  //    final case class AddDouble[Input0 <: Batch]
  //    (
  //      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
  //      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
  //    ) extends DifferentiableArray2D with Cached {
  //
  //      final class Output(val input: Input0, upstream1: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]], upstream2: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]) extends ReferenceCount with Array2DBatch {
  //        type Input >: Input0
  //        val value = upstream1.value.map2(upstream2.value)(_ + _).memoize
  //
  //        override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
  //          upstream1.backward(outputDelta)
  //          upstream2.backward(outputDelta.map {
  //            case None => 0.0
  //            case Some(deltaValue) => deltaValue.sumT
  //          })
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        new Output(input, leftHandSide.forward(input), rightHandSide.forward(input))
  //      }
  //    }
  //
  //    final case class Reciprocal[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Cached with DifferentiableArray2D {
  //
  //      final class Output(val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with Array2DBatch {
  //        type Input >: Input0
  //        val value = upstream.value.map(_ rdiv 1.0).memoize
  //
  //
  //        override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
  //          val upstreamValue = upstream.value
  //          upstream.backward(outputDelta.flatMap[Option[INDArray]] {
  //            case None => Eval.now(None)
  //            case Some(outputDeltaValue) =>
  //              upstreamValue.map {
  //                aValue: INDArray =>
  //                  Some(-outputDeltaValue / (aValue * aValue))
  //              }
  //          }.memoize)
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        val upstream = differentiableArray2D.forward(input)
  //        new Output(input, upstream)
  //      }
  //    }
  //
  //
  //    final case class ReduceSum[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Cached with DifferentiableDouble {
  //
  //      final class Output(val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with DoubleBatch {
  //        type Input >: Input0
  //        val value = upstream.value.map(_.sumT).memoize
  //
  //        override protected def cachedBackward(outputDelta: Eval[scala.Double]): Unit = {
  //          upstream.backward(outputDelta.map2(upstream.value) { (outputDeltaValue, aValue) =>
  //            if (outputDeltaValue == 0) {
  //              None
  //            } else {
  //              Some(Nd4j.valueArrayOf(aValue.shape(), outputDeltaValue))
  //            }
  //          }.memoize)
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        val upstream = differentiableArray2D.forward(input)
  //        new Output(input, upstream)
  //      }
  //    }
  //
  //    final case class Sum[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]], dimensions: Seq[Int]) extends Cached with DifferentiableArray2D {
  //
  //      final class Output(val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with Array2DBatch {
  //        type Input >: Input0
  //        val value = upstream.value.map(_.sum(dimensions: _*)).memoize
  //
  //        override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
  //          val a = upstream.value
  //          upstream.backward(
  //            outputDelta.flatMap[Option[INDArray]] {
  //              case None => Eval.now(None)
  //              case Some(outputDeltaValue) =>
  //                a.map {
  //                  aValue =>
  //                    Some(outputDeltaValue.broadcast(aValue.shape(): _*))
  //                }
  //            }.memoize)
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        val upstream = differentiableArray2D.forward(input)
  //        new Output(input, upstream)
  //      }
  //    }
  //
  //    final case class Negative[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Cached with DifferentiableArray2D {
  //
  //      final class Output(val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with Array2DBatch {
  //        type Input >: Input0
  //        val value = upstream.value.map(-_).memoize
  //
  //        override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
  //          upstream.backward(outputDelta.map {
  //            case None => None
  //            case Some(outputDeltaValue) => Some(-outputDeltaValue)
  //          }.memoize)
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        val upstream = differentiableArray2D.forward(input)
  //        new Output(input, upstream)
  //      }
  //    }
  //
  //    final case class Log[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Cached with DifferentiableArray2D {
  //
  //      final class Output(val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with Array2DBatch {
  //        type Input >: Input0
  //        val value = upstream.value.map(Transforms.log).memoize
  //
  //        override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
  //          val a = upstream.value
  //          upstream.backward(outputDelta.flatMap[Option[INDArray]] {
  //            case None => Eval.now(None)
  //            case Some(outputDeltaValue) => a.map[Option[INDArray]] {
  //              aData: INDArray =>
  //                Some(outputDeltaValue / aData)
  //            }
  //          }.memoize)
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        val upstream = differentiableArray2D.forward(input)
  //        new Output(input, upstream)
  //      }
  //    }
  //
  //    final case class Exp[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Cached with DifferentiableArray2D {
  //
  //      final class Output(val input: Input0, upstream: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]) extends ReferenceCount with Array2DBatch {
  //        type Input >: Input0
  //        val value = upstream.value.map(Transforms.exp).memoize
  //
  //        override protected def cachedBackward(outputDelta: Eval[Option[INDArray]]): Unit = {
  //          upstream.backward(outputDelta.flatMap {
  //            case None => Eval.now(None)
  //            case Some(outputDeltaValue) => value.map {
  //              outputValue: INDArray =>
  //                Some(outputValue * outputDeltaValue)
  //            }
  //          }.memoize)
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        val upstream = differentiableArray2D.forward(input)
  //        new Output(input, upstream)
  //      }
  //    }
  //
  //  }
  //
  //  trait DifferentiableArray2D extends Differentiable {
  //
  //    override type Output <: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]
  //
  //    override def dot(rightHandSide: Array2D) = {
  //      DifferentiableArray2D.Dot[Input](this, rightHandSide)
  //    }
  //
  //    override def +(rightHandSide: Array2D) = {
  //      DifferentiableArray2D.AddArray2D[Input](this, rightHandSide)
  //    }
  //
  //    override def +(rightHandSide: Double) = {
  //      DifferentiableArray2D.AddDouble[Input](this, rightHandSide)
  //    }
  //
  //    override def /(rightHandSide: Array2D) = {
  //      DifferentiableArray2D.MultiplyArray2D[Input](this, DifferentiableArray2D.Reciprocal[Input](rightHandSide))
  //    }
  //
  //    override def /(rightHandSide: Double) = {
  //      DifferentiableArray2D.MultiplyDouble[Input](this, DifferentiableDouble.Reciprocal[Input](rightHandSide))
  //    }
  //
  //    override def *(rightHandSide: Array2D) = {
  //      DifferentiableArray2D.MultiplyArray2D[Input](this, rightHandSide)
  //    }
  //
  //    override def *(rightHandSide: Double) = {
  //      DifferentiableArray2D.MultiplyDouble[Input](this, rightHandSide)
  //    }
  //
  //    override def unary_- = {
  //      DifferentiableArray2D.Negative[Input](this)
  //    }
  //
  //    override def reduceSum = {
  //      DifferentiableArray2D.ReduceSum[Input](this)
  //    }
  //
  //    override def sum(dimensions: Int*) = {
  //      DifferentiableArray2D.Sum[Input](this, dimensions)
  //    }
  //  }
  //
  //  object DifferentiableBoolean {
  //    type Aux[Input0 <: Batch] = DifferentiableBoolean {
  //      type Input = Input0
  //    }
  //
  //    final case class BooleanOps[Input0 <: Batch](generic: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]]) extends DifferentiableBoolean {
  //      override type Output = generic.Output
  //      override type Input = Input0
  //
  //      override def forward(input: Input0): generic.Output = {
  //        generic.forward(input)
  //      }
  //    }
  //
  //    final case class If[Input0 <: Batch, Output0 <: Batch](condition: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]],
  //                                                           `then`: Differentiable.Aux[Input0, Output0],
  //                                                           `else`: Differentiable.Aux[Input0, Output0])
  //      extends Differentiable {
  //      override type Input = Input0
  //      override type Output = Output0
  //
  //      override def forward(input: Input0): Output0 = {
  //        val conditionForwardPass = condition.forward(input)
  //        val output = if (conditionForwardPass.value.value) {
  //          `then`.forward(input)
  //        } else {
  //          `else`.forward(input)
  //        }
  //        conditionForwardPass.backward(Eval.now(false))
  //        output
  //      }
  //    }
  //
  //
  //    final case class Not[Input0 <: Batch](differentiableBoolean: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]]) extends Cached with DifferentiableBoolean {
  //
  //      final class Output(val input: Input0, upstream: Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]) extends ReferenceCount with BooleanBatch {
  //        type Input >: Input0
  //        val value = upstream.value.map(!_)
  //
  //        override protected def cachedBackward(delta: Eval[scala.Boolean]): Unit = {
  //          upstream.backward(delta.map(!_))
  //        }
  //      }
  //
  //      type Input = Input0
  //
  //      override protected def cachedForward(input: Input): Output = {
  //        val upstream = differentiableBoolean.forward(input)
  //        new Output(input, upstream)
  //      }
  //    }
  //
  //  }
  //
  //  trait DifferentiableBoolean extends Differentiable {
  //    type Output <: Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]
  //
  //    override def `if`[A](`then`: A)(`else`: A)(implicit companion: Companion[A]): A = {
  //      companion.specialize(DifferentiableBoolean.If[Input, companion.Output](this, companion.generalize(`then`), companion.generalize(`else`)))
  //    }
  //
  //    override def unary_! : Boolean = {
  //      DifferentiableBoolean.Not[Input](this)
  //    }
  //  }
  //
  //  object DifferentiableHList {
  //    type Aux[Input0 <: Batch] = DifferentiableHList {
  //      type Input = Input0
  //    }
  //  }
  //
  //  trait DifferentiableHList extends Differentiable {
  //    override def ::[Head <: Any : Companion](head: Head): Head :: this.type = ???
  //  }
  //
  //  object DifferentiableHCons {
  //
  //    type Aux[Input0 <: Batch, +Head0, +HeadData0, -HeadDelta0, +Tail0, +TailData0, -TailDelta0] = DifferentiableHCons {
  //      type Input = Input0
  //      type Head <: Head0
  //      type Tail <: Tail0
  //      type HeadData <: HeadData0
  //      type HeadDelta >: HeadDelta0
  //      type TailData <: TailData0
  //      type TailDelta >: TailDelta0
  //    }
  //
  //    final case class HConsOps[Input0 <: Batch, HeadData0, HeadDelta0, TailData0, TailDelta0]
  //    (generic: Differentiable.Aux[Input0, Batch.Aux[
  //      shapeless.::[HeadData0, shapeless.::[TailData0, shapeless.HNil]],
  //      shapeless.::[HeadDelta0, shapeless.::[TailDelta0, shapeless.HNil]]
  //      ]]) extends DifferentiableHCons {
  //      override type Head = Nothing
  //      override type HeadData = HeadData0
  //      override type HeadDelta = HeadDelta0
  //      override type Tail = Nothing
  //      override type TailData = TailData0
  //      override type TailDelta = TailDelta0
  //      override type Output = generic.Output
  //      override type Input = Input0
  //
  //      override def forward(input: Input0): generic.Output = {
  //        generic.forward(input)
  //      }
  //    }
  //
  //  }
  //
  //  trait DifferentiableHCons extends DifferentiableHList {
  //    override type Head <: Differentiable.Aux[Input, Batch.Aux[HeadData, HeadDelta]]
  //    override type Tail <: Differentiable.Aux[Input, Batch.Aux[TailData, TailDelta]]
  //    type HeadData
  //    type HeadDelta
  //    type TailData
  //    type TailDelta
  //    override type Output <: Batch.Aux[
  //      shapeless.::[HeadData, shapeless.::[TailData, shapeless.HNil]],
  //      shapeless.::[HeadDelta, shapeless.::[TailDelta, shapeless.HNil]]
  //      ]
  //
  //    override def head(implicit headCompanion: Companion[Head]): Head = ???
  //
  //    override def tail(implicit tailCompanion: Companion[Tail]): Tail = ???
  //  }
  //
  //  object DifferentiableHNil {
  //
  //    type Aux[Input0 <: Batch] = DifferentiableHNil {
  //      type Input = Input0
  //    }
  //  }
  //
  //  trait DifferentiableHNil extends DifferentiableHList {
  //    override type Output = Batch.Aux[shapeless.HNil, shapeless.HNil]
  //  }
  //


  trait LearningRate {
    def apply(): scala.Double
  }

  object SymbolicDsl {
    type Aux[Input0 <: Batch] = SymbolicDsl {
      type Input = Input0
    }

    //
    //    def apply[Input0 <: Batch](implicit learningRate0: LearningRate) = new SymbolicDsl {
    //      override implicit def learningRate = learningRate0
    //
    //      override type Input = Input0
    //    }
  }

  trait SymbolicDsl extends Dsl {

    trait Any {
      type OutputData <: scala.Any
      type OutputDelta >: scala.Nothing
      val underlying: Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]]
    }

    object Any extends Companion[Any] {
      override type OutputData = scala.Any
      override type OutputDelta = scala.Nothing

      override def fromAst(ast: Any): Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = ast.underlying

      override def toAst(generic: Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) = new Any {

        override type OutputData = scala.Any
        override type OutputDelta = scala.Nothing
        override val underlying = generic
      }
    }

    type Input <: Batch

    trait Companion[Ast <: Any] {

      type OutputData
      type OutputDelta

      def fromAst(ast: Ast): Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]]

      def toAst(generic: Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]]): Ast
    }

    trait HListCompanion[Ast <: HList] extends Companion[Ast] {
      override type OutputData <: shapeless.HList
      override type OutputDelta <: shapeless.HList
    }

    trait HList extends HListApi with Any {

      override type OutputData <: shapeless.HList
      override type OutputDelta <: shapeless.HList

      override def ::[Head <: Any, Tail >: this.type <: HList](head: Head)(implicit headCompanion: Companion[Head], tailCompanion: HListCompanion[Tail]): Head :: Tail = {
        val tail: Tail = this
        val d = DifferentiableHCons.apply[Input, headCompanion.OutputData, headCompanion.OutputDelta, tailCompanion.OutputData, tailCompanion.OutputDelta](
          headCompanion.fromAst(head),
          tailCompanion.fromAst(tail)
        )
        SymbolicDsl.this.::[Head, Tail].toAst(d)
      }
    }

    object HList extends HListCompanion[HList] {

      override type OutputData = shapeless.HList
      override type OutputDelta = scala.Nothing

      override def fromAst(ast: HList) = ast.underlying

      override def toAst(generic: Differentiable.Aux[Input, Batch.Aux[OutputData, scala.Nothing]]): HList = new HList {
        override type OutputData = shapeless.HList
        override type OutputDelta = scala.Nothing
        override val underlying = generic
      }
    }

    trait HNil extends HList {
      override type OutputData = shapeless.HNil
      override type OutputDelta = shapeless.HNil
    }

    object HNil extends HNil with HListCompanion[HNil] {
      override def toAst(generic: Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) = this

      override val underlying = DifferentiableHNil[Input]()

      override def fromAst(ast: HNil) = ast.underlying
    }

    sealed trait ::[Head <: Any, Tail <: HList] extends HList with HConsApi[Head, Tail] {

      type HeadData
      type HeadDelta
      type TailData <: shapeless.HList
      type TailDelta <: shapeless.HList

      override type OutputData = shapeless.::[HeadData, TailData]
      override type OutputDelta = shapeless.::[HeadDelta, TailDelta]

      def head: Head = ???

      def tail: Tail = ???

      val underlying: Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]]

    }


    override implicit def ::[Head <: Any, Tail <: HList](implicit headCompanion0: Companion[Head], tailCompanion0: HListCompanion[Tail]) = {
      new HListCompanion[Head :: Tail] {

        override type OutputData = shapeless.::[headCompanion0.OutputData, tailCompanion0.OutputData]
        override type OutputDelta = shapeless.::[headCompanion0.OutputDelta, tailCompanion0.OutputDelta]

        override def toAst(generic: Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) = new (Head :: Tail) {
          override type HeadData = headCompanion0.OutputData
          override type HeadDelta = headCompanion0.OutputDelta
          override type TailData = tailCompanion0.OutputData
          override type TailDelta = tailCompanion0.OutputDelta
          override val underlying = generic
        }

        override def fromAst(ast: Head :: Tail): Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = {
          ast.asInstanceOf[(Head :: Tail) {
            type HeadData = headCompanion0.OutputData
            type HeadDelta = headCompanion0.OutputDelta
            type TailData = tailCompanion0.OutputData
            type TailDelta = tailCompanion0.OutputDelta
          }].underlying
        }
      }
    }


    //    override type HList = this.type

    //
    //    override type Any = Differentiable.Aux[Input, Batch.Aux[scala.Any, scala.Nothing]]
    //
    //    override object Any extends Specialize {
    //      override type SpecialDifferentiable = Any
    //      override type Input = SymbolicDsl.this.Input
    //      override type OutputData = scala.Any
    //      override type OutputDelta = scala.Nothing
    //
    //      override def specialize(generic: Any): Any = generic
    //
    //      /**
    //        * Returns the base [[Differentiable]] type of a [[SpecialDifferentiable]].
    //        */
    //      override def generalize = implicitly
    //    }
    //
    //    override type Double = DifferentiableDouble.Aux[Input]
    //
    //    implicit override object Double extends Dsl.Lifter with Specialize with (scala.Double => Double) {
    //
    //      import DifferentiableDouble._
    //
    //      override type LiftFrom = scala.Double
    //      override type LiftTo = Double
    //      override type SpecialDifferentiable = Double
    //      override type Input = SymbolicDsl.this.Input
    //      override type OutputData = Eval[scala.Double]
    //      override type OutputDelta = Eval[scala.Double]
    //
    //      override def apply(value: scala.Double) = DoubleLiteral[Input](value)
    //
    //      override def weight(initialValue: scala.Double) = DoubleWeight[Input](initialValue)
    //
    //      override def specialize(generic: Differentiable.Aux[Input, Output]) = DoubleOps(generic)
    //
    //      override def generalize = implicitly
    //
    //    }
    //
    //    override type Array2D = DifferentiableArray2D.Aux[Input]
    //
    //    implicit override object Array2D extends Specialize with Array2DCompanionApi {
    //
    //      override type SpecialDifferentiable = Array2D
    //      override type Input = SymbolicDsl.this.Input
    //      override type OutputData = Eval[INDArray]
    //      override type OutputDelta = Eval[Option[INDArray]]
    //
    //      override def apply(value: Array[Array[scala.Double]]) = DifferentiableArray2D.Array2DLiteral[Input](value)
    //
    //      override def weight(initialValue: Array[Array[scala.Double]]) = DifferentiableArray2D.Array2DWeight[Input](initialValue)
    //
    //      override def generalize = implicitly
    //
    //      override def specialize(generic: Differentiable.Aux[Input, Output]) = DifferentiableArray2D.Array2DOps(generic)
    //
    //      override def randn(numberOfRows: Int, numberOfColumns: Int): Array2D = DifferentiableArray2D.Array2DWeight.randn(numberOfRows, numberOfColumns)
    //
    //      override def zeros(numberOfRows: Int, numberOfColumns: Int): Array2D = DifferentiableArray2D.Array2DWeight.zeros(numberOfRows, numberOfColumns)
    //    }
    //
    //    override type Boolean = DifferentiableBoolean.Aux[Input]
    //
    //    implicit override object Boolean extends Specialize {
    //
    //      import DifferentiableBoolean._
    //
    //      override type SpecialDifferentiable = Boolean
    //      override type Input = SymbolicDsl.this.Input
    //      override type OutputData = Eval[scala.Boolean]
    //      override type OutputDelta = Eval[scala.Boolean]
    //
    //      override def generalize = implicitly
    //
    //      override def specialize(generic: Differentiable.Aux[Input, Output]) = BooleanOps(generic)
    //
    //    }
    //
    //    override def exp(array: Array2D): Array2D = {
    //      DifferentiableArray2D.Exp[Input](array)
    //    }
    //
    //    override def log(array: Array2D): Array2D = {
    //      DifferentiableArray2D.Log[Input](array)
    //    }
    //
    //    override def max(leftHandSide: Array2D, rightHandSide: Double) = {
    //      DifferentiableArray2D.MaxDouble(leftHandSide, rightHandSide)
    //    }
    //
    //    override type ::[+Head, +Tail] = DifferentiableHCons.Aux[
    //      Input,
    //      Head, _, _,
    //      Tail, _, _
    //      ]
    //
    //    override implicit def ::[Head <: Any, Tail <: HList](implicit headCompanion: Companion[Head], tailCompanion: Companion[Tail]): Companion[Head :: Tail] = {
    //      val hconCompanion: Companion[DifferentiableHCons.Aux[
    //        Input,
    //        headCompanion.SpecialDifferentiable, headCompanion.OutputData, headCompanion.OutputDelta,
    //        tailCompanion.SpecialDifferentiable, tailCompanion.OutputData, tailCompanion.OutputDelta
    //        ]]
    //      = new Specialize {
    //        override type Input = SymbolicDsl.this.Input
    //        override type OutputData = shapeless.::[headCompanion.OutputData, shapeless.::[tailCompanion.OutputData, shapeless.HNil]]
    //        override type OutputDelta = shapeless.::[headCompanion.OutputDelta, shapeless.::[tailCompanion.OutputDelta, shapeless.HNil]]
    //        override type SpecialDifferentiable =
    //        DifferentiableHCons.Aux[
    //          Input,
    //          headCompanion.SpecialDifferentiable, headCompanion.OutputData, headCompanion.OutputDelta,
    //          tailCompanion.SpecialDifferentiable, tailCompanion.OutputData, tailCompanion.OutputDelta
    //          ]
    //
    //        override def generalize = implicitly
    //
    //        /**
    //          * Returns a special subclass of a [[Differentiable]].
    //          */
    //        override def specialize(generic: Differentiable.Aux[Input, Output]) = {
    //          DifferentiableHCons.HConsOps[Input, headCompanion.OutputData, headCompanion.OutputDelta, tailCompanion.OutputData, tailCompanion.OutputDelta](generic)
    //        }
    //      }
    //
    //      // FIXME: I don't know how to prove:
    //      //
    //      // (DifferentiableHCons.Aux[Input, headCompanion.SpecialDifferentiable, _, _, tailCompanion.SpecialDifferentiable, _, _])
    //      // =:=
    //      // (DifferentiableHCons.Aux[Input, headCompanion.SpecialDifferentiable, headCompanion.OutputData, headCompanion.OutputDelta, tailCompanion.SpecialDifferentiable, tailCompanion.OutputData, tailCompanion.OutputDelta])
    //      hconCompanion.asInstanceOf[Companion[DifferentiableHCons.Aux[
    //        Input,
    //        headCompanion.SpecialDifferentiable, _, _,
    //        tailCompanion.SpecialDifferentiable, _, _
    //        ]]]
    //    }
    //
    //    type HList = DifferentiableHList.Aux[Input]
    //
    //    type HNil = DifferentiableHNil.Aux[Input]
    //
    //    override implicit object HNil extends Specialize with DifferentiableHNil {
    //      type SpecialDifferentiable = HNil
    //      type Input = SymbolicDsl.this.Input
    //      override type OutputData = shapeless.HNil
    //      override type OutputDelta = shapeless.HNil
    //      override type Output = Batch.Aux[shapeless.HNil, shapeless.HNil]
    //
    //      override def specialize(generic: Differentiable.Aux[Input, Output]) = this
    //
    //      override def forward(input: Input): Output = HNilBatch
    //
    //      override def generalize = implicitly
    //    }
    //
  }


}

trait Differentiable {

  import Differentiable._

  //  type Companion[SpecialDifferentiable0] = Specialize {
  //    type SpecialDifferentiable = SpecialDifferentiable0
  //    type Input = Differentiable.this.Input
  //  }
  //  type Any = Differentiable.Aux[Input, Batch.Aux[scala.Any, scala.Nothing]]
  //  type Array2D = DifferentiableArray2D.Aux[Input]
  //  type Double = DifferentiableDouble.Aux[Input]
  //  type Boolean = DifferentiableBoolean.Aux[Input]
  //  type HList = DifferentiableHList.Aux[Input]
  //  type ::[+Head, +Tail] = DifferentiableHCons.Aux[Input, Head, _, _, Tail, _, _]
  type Input <: Batch

  type Output <: Batch.Aux[scala.Any, scala.Nothing]

  def forward(input: Input): Output

}