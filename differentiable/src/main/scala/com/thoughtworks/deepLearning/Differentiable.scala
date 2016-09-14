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


object Differentiable {

  object Batch {
    type Aux[+Data0, -Delta0] = Batch {
      type Data <: Data0
      type Delta >: Delta0
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

  type Aux[-Input0 <: Batch, +Output0 <: Batch.Aux[_, _]] = Differentiable {
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

  object DifferentiableDouble {
    type Aux[Input0] = DifferentiableDouble {
      type Input = Input0
    }

    final case class LessThan[Input0 <: Batch]
    (
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
    ) extends Cached with DifferentiableBoolean {

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

    final case class Add[Input0 <: Batch]
    (
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
    ) extends Cached with DifferentiableDouble {

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

    final case class Substract[Input0 <: Batch]
    (
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
    ) extends Cached with DifferentiableDouble {

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

    final case class Multiply[Input0 <: Batch]
    (
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
    ) extends Cached with DifferentiableDouble {

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

    final case class Reciprocal[Input0 <: Batch](differentiableDouble: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]) extends Cached with DifferentiableDouble {

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

    final case class Negative[Input0 <: Batch](differentiableDouble: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]) extends Cached with DifferentiableDouble {

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

    final case class DoubleLiteral[Input0 <: Batch](rawValue: scala.Double) extends DifferentiableDouble with DoubleBatch {
      override type Input = Input0
      override type Output = DoubleLiteral[Input0]

      override def forward(any: Input) = this

      override def backward(delta: Delta): Unit = {}

      override def value = Eval.now(rawValue)
    }

    final case class DoubleWeight[Input0 <: Batch](var rawValue: scala.Double)(implicit learningRate: LearningRate) extends DifferentiableDouble with DoubleBatch {
      override type Input = Input0
      override type Output = DoubleWeight[Input0]

      override def forward(any: Input) = this

      override def backward(delta: Delta): Unit = {
        rawValue -= delta.value * learningRate()
      }

      override def value = Eval.now(rawValue)

    }

    final case class DoubleOps[Input0 <: Batch](generic: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]) extends DifferentiableDouble {
      override type Output = generic.Output
      override type Input = Input0

      override def forward(input: Input0): generic.Output = {
        generic.forward(input)
      }
    }

  }

  trait DifferentiableDouble extends Differentiable with Dsl.DoubleApi {

    import DifferentiableDouble._

    override type Output <: Batch.Aux[Eval[scala.Double], Eval[scala.Double]]

    override def unary_- = {
      Negative[Input](this)
    }

    override def -(rightHandSide: Double) = {
      Substract[Input](this, rightHandSide)
    }

    override def <(rightHandSide: Double) = {
      LessThan[Input](this, rightHandSide)
    }

    override def +(rightHandSide: Double) = {
      Add[Input](this, rightHandSide)
    }

    override def /(rightHandSide: Double) = {
      Multiply[Input](this, Reciprocal[Input](rightHandSide))
    }

    override def /(rightHandSide: Array2D) = {
      DifferentiableArray2D.MultiplyDouble[Input](DifferentiableArray2D.Reciprocal[Input](rightHandSide), this)
    }

    override def *(rightHandSide: Double) = {
      Multiply[Input](this, rightHandSide)
    }
  }

  object DifferentiableArray2D {
    type Aux[Input0] = DifferentiableArray2D {
      type Input = Input0
    }

    final case class Array2DLiteral[Input0 <: Batch](rawValue: INDArray) extends DifferentiableArray2D with Array2DBatch {
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

    final case class Array2DWeight[Input0 <: Batch](var rawValue: INDArray)(implicit learningRate: LearningRate) extends DifferentiableArray2D with Array2DBatch {
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
      def apply[Input <: Batch](nativeArray: Array[Array[scala.Double]])(implicit learningRate: LearningRate): Array2DWeight[Input] = {
        new Array2DWeight[Input](nativeArray.toNDArray)
      }
    }

    final case class Array2DOps[Input0 <: Batch](generic: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends DifferentiableArray2D {
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
    ) extends DifferentiableArray2D with Cached {

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

    final case class MultiplyArray2D[Input0 <: Batch]
    (
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]
    ) extends DifferentiableArray2D with Cached {

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

    final case class AddArray2D[Input0 <: Batch]
    (
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]
    ) extends DifferentiableArray2D with Cached {

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


    final case class MultiplyDouble[Input0 <: Batch]
    (
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
    ) extends DifferentiableArray2D with Cached {

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

    final case class AddDouble[Input0 <: Batch]
    (
      leftHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]],
      rightHandSide: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
    ) extends DifferentiableArray2D with Cached {

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

    final case class Reciprocal[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Cached with DifferentiableArray2D {

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


    final case class ReduceSum[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Cached with DifferentiableDouble {

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

    final case class Sum[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]], dimensions: Seq[Int]) extends Cached with DifferentiableArray2D {

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

    final case class Negative[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Cached with DifferentiableArray2D {

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

    final case class Log[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Cached with DifferentiableArray2D {

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

    final case class Exp[Input0 <: Batch](differentiableArray2D: Differentiable.Aux[Input0, Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]) extends Cached with DifferentiableArray2D {

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

  }

  trait DifferentiableArray2D extends Differentiable with Dsl.Array2DApi {

    override type Output <: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]

    override def dot(rightHandSide: Array2D) = {
      DifferentiableArray2D.Dot[Input](this, rightHandSide)
    }

    override def +(rightHandSide: Array2D) = {
      DifferentiableArray2D.AddArray2D[Input](this, rightHandSide)
    }

    override def +(rightHandSide: Double) = {
      DifferentiableArray2D.AddDouble[Input](this, rightHandSide)
    }

    override def /(rightHandSide: Array2D) = {
      DifferentiableArray2D.MultiplyArray2D[Input](this, DifferentiableArray2D.Reciprocal[Input](rightHandSide))
    }

    override def /(rightHandSide: Double) = {
      DifferentiableArray2D.MultiplyDouble[Input](this, DifferentiableDouble.Reciprocal[Input](rightHandSide))
    }

    override def *(rightHandSide: Array2D) = {
      DifferentiableArray2D.MultiplyArray2D[Input](this, rightHandSide)
    }

    override def *(rightHandSide: Double) = {
      DifferentiableArray2D.MultiplyDouble[Input](this, rightHandSide)
    }

    override def unary_- = {
      DifferentiableArray2D.Negative[Input](this)
    }

    override def reduceSum = {
      DifferentiableArray2D.ReduceSum[Input](this)
    }

    override def sum(dimensions: Int*) = {
      DifferentiableArray2D.Sum[Input](this, dimensions)
    }
  }

  object DifferentiableBoolean {
    type Aux[Input0] = DifferentiableBoolean {
      type Input = Input0
    }

    final case class BooleanOps[Input0 <: Batch](generic: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]]) extends DifferentiableBoolean {
      override type Output = generic.Output
      override type Input = Input0

      override def forward(input: Input0): generic.Output = {
        generic.forward(input)
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


    final case class Not[Input0 <: Batch](differentiableBoolean: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]]) extends Cached with DifferentiableBoolean {

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

  }

  trait DifferentiableBoolean extends Differentiable with Dsl.BooleanApi {
    type Output <: Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]

    override def `if`[A](`then`: A)(`else`: A)(implicit companion: Companion[A]): A = {
      companion.specialize(DifferentiableBoolean.If[Input, companion.Output](this, companion.generalize(`then`), companion.generalize(`else`)))
    }

    override def unary_! : Boolean = {
      DifferentiableBoolean.Not[Input](this)
    }
  }

  object Specialize {
    type Aux[Input0, Output0, SpecialDifferentiable0] = Specialize {
      type Input = Input0
      type Output = Output0
      type SpecialDifferentiable = SpecialDifferentiable0
    }
  }

  trait Specialize {
    type Input <: Batch
    type Output <: Batch
    type SpecialDifferentiable

    /**
      * Returns the base [[Differentiable]] type of a [[SpecialDifferentiable]].
      */
    def generalize(specialFunction: SpecialDifferentiable): Differentiable.Aux[Input, Output]

    /**
      * Returns a special subclass of a [[Differentiable]].
      */
    def specialize(generic: Differentiable.Aux[Input, Output]): SpecialDifferentiable
  }

  trait LearningRate {
    def apply(): scala.Double
  }

  object SymbolicDsl {
    type Aux[Input0 <: Batch] = SymbolicDsl {
      type Input = Input0
    }

    def apply[Input0 <: Batch](implicit learningRate0: LearningRate) = new SymbolicDsl {
      override implicit def learningRate = learningRate0
      override type Input = Input0
    }
  }

  trait SymbolicDsl extends Dsl {

    type Input <: Batch

    implicit def learningRate: LearningRate

    override type Companion[SpecialDifferentiable0] = Specialize {
      type SpecialDifferentiable = SpecialDifferentiable0
      type Input = SymbolicDsl.this.Input
    }

    override type Any = Differentiable.Aux[Input, Batch.Aux[_, _]]

    override object Any extends Specialize {
      override type SpecialDifferentiable = Any
      override type Input = SymbolicDsl.this.Input
      override type Output = Batch.Aux[_, _]

      override def generalize(generic: Any): Any = generic

      override def specialize(generic: Any): Any = generic
    }

    override type Double = DifferentiableDouble.Aux[Input]

    implicit override object Double extends Dsl.Lifter with Specialize {

      import DifferentiableDouble._

      override type LiftFrom = scala.Double
      override type LiftTo = Double
      override type SpecialDifferentiable = Double
      override type Input = SymbolicDsl.this.Input
      override type Output = Batch.Aux[Eval[scala.Double], Eval[scala.Double]]

      override def apply(value: scala.Double) = DoubleLiteral[Input](value)

      override def weight(initialValue: scala.Double) = DoubleWeight[Input](initialValue)

      override def specialize(generic: Differentiable.Aux[Input, Output]) = DoubleOps(generic)

      override def generalize(generic: Double): Double = generic

    }

    override type Array2D = DifferentiableArray2D.Aux[Input]

    implicit override object Array2D extends Dsl.Lifter with Specialize {

      import DifferentiableArray2D._

      override type LiftFrom = Array[Array[scala.Double]]
      override type LiftTo = Array2D
      override type SpecialDifferentiable = Array2D
      override type Input = SymbolicDsl.this.Input
      override type Output = Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]

      override def apply(value: Array[Array[scala.Double]]) = DifferentiableArray2D.Array2DLiteral[Input](value)

      override def weight(initialValue: Array[Array[scala.Double]]) = Array2DWeight[Input](initialValue)

      override def generalize(specialFunction: Array2D) = specialFunction

      override def specialize(generic: Differentiable.Aux[Input, Output]) = Array2DOps(generic)

    }

    override type Boolean = DifferentiableBoolean.Aux[Input]

    implicit override object Boolean extends Specialize {

      import DifferentiableBoolean._

      override type SpecialDifferentiable = Boolean
      override type Input = SymbolicDsl.this.Input
      override type Output = Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]

      override def generalize(generic: Boolean): Boolean = generic

      override def specialize(generic: Differentiable.Aux[Input, Output]) = BooleanOps(generic)

    }

    override def exp(array: Array2D): Array2D = {
      DifferentiableArray2D.Exp[Input](array)
    }

    override def log(array: Array2D): Array2D = {
      DifferentiableArray2D.Log[Input](array)
    }
  }


}

trait Differentiable {

  import Differentiable._

  type Companion[SpecialDifferentiable0] = Specialize {
    type SpecialDifferentiable = SpecialDifferentiable0
    type Input = Differentiable.this.Input
  }
  type Array2D = DifferentiableArray2D.Aux[Input]
  type Double = DifferentiableDouble.Aux[Input]
  type Boolean = DifferentiableBoolean.Aux[Input]

  type Input <: Batch

  type Output <: Batch.Aux[_, _]

  def forward(input: Input): Output

}