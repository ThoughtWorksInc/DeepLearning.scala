package com.thoughtworks.deeplearning

import cats.implicits._
import cats.{Applicative, Eval, Semigroup, Traverse}
import com.thoughtworks.deeplearning.Layer.{Aux, Batch, CloseableOnce}
import com.thoughtworks.deeplearning.Conversion._
import com.thoughtworks.deeplearning.Bp2DArray.Layers._
import com.thoughtworks.deeplearning.Bp2DArray.Optimizers._
import com.thoughtworks.deeplearning.BpDouble._
import com.thoughtworks.deeplearning.Conversion.Layers.Literal
import com.thoughtworks.deeplearning.Layer.Batch.Aux
import org.nd4j.linalg.api.ndarray.INDArray
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import com.thoughtworks.deeplearning.Poly.MathMethods
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.ops.transforms.Transforms.sign
import org.nd4s.Implicits._

import language.higherKinds
import language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object Bp2DArray {

  private[deeplearning] trait TwoDArraySemigroupBatch extends Batch {

    override type Data = Eval[INDArray]

    override type Delta = Eval[INDArray]

    protected final def semigroup = new Semigroup[Delta] {
      override def combine(x: Delta, y: Delta): Delta = x.map2(y)(_ + _)
    }

  }

  private[Bp2DArray] def sumAs(outputDeltaValue: INDArray, shape: Array[Int]) = {
    shape match {
      case Array(1, 1) => outputDeltaValue.sum(0, 1)
      case Array(_, 1) => outputDeltaValue.sum(1)
      case Array(1, _) => outputDeltaValue.sum(0)
      case Array(_, _) => outputDeltaValue
    }
  }

  /** @template */
  type Bp2DArray = BackPropagationType[Eval[INDArray], Eval[INDArray]]

  object Optimizers {

    trait L1Regularization extends LearningRate {
      protected def l1Regularization: Double

      override def updateNDArray(oldValue: INDArray, delta: INDArray): INDArray = {
        super.updateNDArray(oldValue, delta) - sign(oldValue) * l1Regularization * currentLearningRate()
      }

    }

    trait L2Regularization extends LearningRate {
      protected def l2Regularization: Double

      override def updateNDArray(oldValue: INDArray, delta: INDArray): INDArray = {
        super.updateNDArray(oldValue, delta) - oldValue * l2Regularization * currentLearningRate()
      }

    }

    trait Optimizer {
      def updateNDArray(oldValue: INDArray, delta: INDArray): INDArray
    }

    trait LearningRate extends Optimizer {

      protected def currentLearningRate(): Double

      override def updateNDArray(oldValue: INDArray, delta: INDArray): INDArray = {
        oldValue - delta * currentLearningRate()
      }
    }

  }

  import Optimizers._

  object Layers {

    final case class MultiplyBp2DArray[Input0 <: Batch](
        operand1: Layer.Aux[Input0, Bp2DArray#Batch],
        operand2: Layer.Aux[Input0, Bp2DArray#Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch {

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

          override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
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
      }
    }

    final case class MaxBpDouble[Input0 <: Batch](
        operand1: Layer.Aux[Input0, Bp2DArray#Batch],
        operand2: Layer.Aux[Input0, BpDouble#Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch {

          val value = upstream1.value.map2(upstream2.value)(Transforms.max).memoize

          override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
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
      }
    }

    final case class PlusBp2DArray[Input0 <: Batch](
        operand1: Layer.Aux[Input0, Bp2DArray#Batch],
        operand2: Layer.Aux[Input0, Bp2DArray#Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch {

          val value = {
            upstream1.value
              .map2(upstream2.value) { (aValue, bValue) =>
                val Array(aRows, aColumns) = aValue.shape()
                val Array(bRows, bColumns) = bValue.shape()
                val newShape =
                  Array(math.max(aRows, bRows), math.max(aColumns, bColumns))
                aValue.broadcast(newShape: _*) + bValue.broadcast(newShape: _*)
              }
              .memoize
          }

          override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
            val sumAsOriginalShape = { (outputDeltaValue: INDArray, upstreamValue: INDArray) =>
              sumAs(outputDeltaValue, upstreamValue.shape)
            }
            upstream1.backward(outputDelta.map2(upstream1.value)(sumAsOriginalShape))
            upstream2.backward(outputDelta.map2(upstream2.value)(sumAsOriginalShape))
          }
        }
      }
    }

    object ToBp2DArray {

      private[ToBp2DArray] implicit object SeqInstances extends Traverse[Seq] {
        override def traverse[G[_]: Applicative, A, B](fa: Seq[A])(f: (A) => G[B]): G[Seq[B]] = {
          fa.foldRight((Vector.empty[B]: Seq[B]).pure[G]) { (a: A, accumulation: G[Seq[B]]) =>
            f(a).map2(accumulation)(_ +: _)
          }
        }

        override def foldLeft[A, B](fa: Seq[A], b: B)(f: (B, A) => B): B = {
          fa.foldLeft(b)(f)
        }

        override def foldRight[A, B](fa: Seq[A], lb: Eval[B])(f: (A, Eval[B]) => Eval[B]): Eval[B] = {
          fa.foldRight(lb)(f)
        }
      }

    }

    object ToSeq {
      private[ToSeq] trait Seq2DBatch extends Batch {
        override type Data = Seq[Seq[Eval[Double]]]
        override type Delta = (Int, (Int, Eval[Double]))
      }
    }

    /**
      * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
      */
    final case class ToSeq[Input0 <: Batch](operand: Layer.Aux[Input0, Bp2DArray#Batch]) extends BufferedLayer.Unary {
      import ToSeq._
      type BufferedBatch =
        UnaryBatch with Seq2DBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with UnaryBatch with Seq2DBatch {

          private def zeroDelta =
            upstream.value.map { upstreamData =>
              Nd4j.zeros(upstreamData.shape: _*)
            }.memoize

          @volatile
          var upstreamDelta = zeroDelta

          override protected def flush(): Unit = {
            upstream.backward(synchronized {
              val oldDelta = upstreamDelta
              upstreamDelta = zeroDelta
              oldDelta
            })
          }

          override def backward(delta: Delta): Unit = {
            synchronized {
              val (i, (j, value)) = delta
              // Cannot use += because of https://issues.scala-lang.org/browse/SI-10021
              upstreamDelta.value(i, j) = upstreamDelta.value(i, j) + value.value
            }
          }

          override val value: Data = {
            val ndarray = upstream.value.value
            val doubleArray = ndarray.data.asDouble()
            for (i <- (0 until ndarray.rows).view) yield {
              doubleArray.view(i * ndarray.columns, (i + 1) * ndarray.columns).map { doubleValue =>
                Eval.now(doubleValue)
              }
            }
          }
        }
      }
    }

    final case class Weight(var rawValue: INDArray)(implicit optimizer: Optimizer)
        extends Layer
        with TwoDArraySemigroupBatch {
      override type Input = Batch
      override type Output = Batch.Aux[Data, Delta]

      override def addReference() = this

      override def value = Eval.now(rawValue)

      override def forward(any: Input) = this

      override def backward(delta: Delta): Unit = {
        synchronized {
          rawValue = optimizer.updateNDArray(rawValue, delta.value)
        }
      }

      override def close(): Unit = {}

    }

    final case class ToBp2DArray[Input0 <: Batch](
        operands: Seq[Seq[Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]]]])
        extends Layer {

      import ToBp2DArray.SeqInstances

      type Input = Input0

      final class Output private[ToBp2DArray] (upstreams: Seq[Seq[Batch.Aux[Eval[Double], Eval[Double]]]])
          extends TwoDArraySemigroupBatch
          with CloseableOnce {
        override def backward(delta: Eval[INDArray]): Unit = {
          for ((row, i) <- upstreams.view.zipWithIndex; (upstream, j) <- row.zipWithIndex) {
            upstream.backward(delta.map(_(i, j)))
          }

        }

        override val value = {
          upstreams.traverse(_.traverse(_.value)).map(_.toNDArray).memoize
        }

        override def close(): Unit = {
          super.close()
          upstreams.foreach(_.foreach(_.close()))
        }

        override def addReference(): Output = {
          new Output(upstreams.map(_.map(_.addReference())))
        }
      }

      override def forward(input: Input) = {
        new Output(operands.map(_.map(_.forward(input))))
      }
    }

    final case class Sum[Input0 <: Batch](operand: Layer.Aux[Input0, Bp2DArray#Batch], dimensions: Seq[Int])
        extends BufferedLayer.Unary {
      type BufferedBatch = TwoDArraySemigroupBatch with SemigroupBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with TwoDArraySemigroupBatch with SemigroupBatch with UnaryBatch {

          val value = upstream.value.map(_.sum(dimensions: _*)).memoize

          override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
            val a = upstream.value
            upstream.backward(
              outputDelta
                .map2(a) { (outputDeltaValue, aValue) =>
                  outputDeltaValue.broadcast(aValue.shape: _*)
                }
                .memoize)
          }
        }
      }
    }

    final case class ReduceSum[Input0 <: Batch](operand: Layer.Aux[Input0, Bp2DArray#Batch])
        extends BufferedLayer.Unary {
      type BufferedBatch = DoubleMonoidBatch with MonoidBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with DoubleMonoidBatch with MonoidBatch with UnaryBatch {

          val value = upstream.value.map(_.sumT).memoize

          override protected def rawBackward(outputDelta: Eval[Double]): Unit = {
            upstream.backward(
              outputDelta
                .map2(upstream.value) { (outputDeltaValue, aValue) =>
                  Nd4j.valueArrayOf(aValue.shape(), outputDeltaValue)
                }
                .memoize)
          }
        }
      }
    }

    final case class Reciprocal[Input0 <: Batch](operand: Layer.Aux[Input0, Bp2DArray#Batch])
        extends BufferedLayer.Unary {
      type BufferedBatch = TwoDArraySemigroupBatch with SemigroupBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with TwoDArraySemigroupBatch with SemigroupBatch with UnaryBatch {

          val value = upstream.value.map(_ rdiv 1.0).memoize

          override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
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
      }
    }

    final case class PlusBpDouble[Input0 <: Batch](
        operand1: Layer.Aux[Input0, Bp2DArray#Batch],
        operand2: Layer.Aux[Input0, BpDouble#Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch {
          val value = upstream1.value.map2(upstream2.value)(_ + _).memoize

          override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
            upstream1.backward(outputDelta)
            upstream2.backward(outputDelta.map(_.sumT))
          }
        }
      }
    }

    final case class Negative[Input0 <: Batch](operand: Layer.Aux[Input0, Bp2DArray#Batch])
        extends BufferedLayer.Unary {
      type BufferedBatch = TwoDArraySemigroupBatch with SemigroupBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with TwoDArraySemigroupBatch with SemigroupBatch with UnaryBatch {

          val value = upstream.value.map(-_).memoize

          override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
            upstream.backward(outputDelta.map(-_).memoize)
          }
        }
      }
    }

    final case class Exp[Input0 <: Batch](operand: Layer.Aux[Input0, Bp2DArray#Batch]) extends BufferedLayer.Unary {
      type BufferedBatch = TwoDArraySemigroupBatch with SemigroupBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with TwoDArraySemigroupBatch with SemigroupBatch with UnaryBatch {
          val value = upstream.value.map(Transforms.exp).memoize

          override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
            upstream.backward(value.map2(outputDelta)(_ * _).memoize)
          }
        }
      }
    }

    final case class Log[Input0 <: Batch](operand: Layer.Aux[Input0, Bp2DArray#Batch]) extends BufferedLayer.Unary {
      type BufferedBatch = TwoDArraySemigroupBatch with SemigroupBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with TwoDArraySemigroupBatch with SemigroupBatch with UnaryBatch {

          val value = upstream.value.map(Transforms.log).memoize

          override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
            upstream.backward(outputDelta.map2(upstream.value)(_ / _).memoize)
          }
        }
      }
    }

    final case class MultiplyBpDouble[Input0 <: Batch](
        operand1: Layer.Aux[Input0, Bp2DArray#Batch],
        operand2: Layer.Aux[Input0, BpDouble#Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch {

          val value = upstream1.value.map2(upstream2.value)(_ * _).memoize

          override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
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
      }
    }

    final case class Dot[Input0 <: Batch](
        operand1: Layer.Aux[Input0, Bp2DArray#Batch],
        operand2: Layer.Aux[Input0, Bp2DArray#Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch {

          override val value = upstream1.value.map2(upstream2.value)(_ dot _).memoize

          override protected def rawBackward(outputDelta: Eval[INDArray]): Unit = {
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

      }
    }

  }

  import Layers._

  implicit def `max(Bp2DArray,Double)`[Left, Right, Input <: Batch]: max.Case.Aux[Layer.Aux[Input, Bp2DArray#Batch],
                                                                                  Layer.Aux[Input, BpDouble#Batch],
                                                                                  Layer.Aux[Input, Bp2DArray#Batch]] =
    max.at(MaxBpDouble(_, _))

  implicit def `Bp2DArray/Bp2DArray`[Input <: Batch]: MathMethods./.Case.Aux[Layer.Aux[Input, Bp2DArray#Batch],
                                                                             Layer.Aux[Input, Bp2DArray#Batch],
                                                                             Layer.Aux[Input, Bp2DArray#Batch]] = {
    MathMethods./.at { (leftLayer, rightLayer) =>
      MultiplyBp2DArray(leftLayer, Reciprocal(rightLayer))
    }
  }

  implicit def `BpDouble/Bp2DArray`[Input <: Batch]: MathMethods./.Case.Aux[Layer.Aux[Input, BpDouble#Batch],
                                                                            Layer.Aux[Input, Bp2DArray#Batch],
                                                                            Layer.Aux[Input, Bp2DArray#Batch]] = {
    MathMethods./.at { (leftLayer, rightLayer) =>
      MultiplyBpDouble(Reciprocal(rightLayer), leftLayer)
    }
  }

  implicit def `Bp2DArray/BpDouble`[Input <: Batch]: MathMethods./.Case.Aux[Layer.Aux[Input, Bp2DArray#Batch],
                                                                            Layer.Aux[Input, BpDouble#Batch],
                                                                            Layer.Aux[Input, Bp2DArray#Batch]] = {
    MathMethods./.at { (leftLayer, rightLayer) =>
      MultiplyBpDouble(leftLayer, BpDouble.Layers.Reciprocal(rightLayer))
    }
  }

  implicit def `Bp2DArray*Bp2DArray`[Input <: Batch]: MathMethods.*.Case.Aux[Layer.Aux[Input, Bp2DArray#Batch],
                                                                             Layer.Aux[Input, Bp2DArray#Batch],
                                                                             Layer.Aux[Input, Bp2DArray#Batch]] = {
    MathMethods.*.at { (leftLayer, rightLayer) =>
      MultiplyBp2DArray(leftLayer, rightLayer)
    }
  }

  implicit def `Bp2DArray*BpDouble`[Input <: Batch]: MathMethods.*.Case.Aux[Layer.Aux[Input, Bp2DArray#Batch],
                                                                            Layer.Aux[Input, BpDouble#Batch],
                                                                            Layer.Aux[Input, Bp2DArray#Batch]] = {
    MathMethods.*.at { (leftLayer, rightLayer) =>
      MultiplyBpDouble(leftLayer, rightLayer)
    }
  }

  implicit def `BpDouble*Bp2DArray`[Input <: Batch]: MathMethods.*.Case.Aux[Layer.Aux[Input, BpDouble#Batch],
                                                                            Layer.Aux[Input, Bp2DArray#Batch],
                                                                            Layer.Aux[Input, Bp2DArray#Batch]] = {
    MathMethods.*.at { (leftLayer, rightLayer) =>
      MultiplyBpDouble(rightLayer, leftLayer)
    }
  }

  implicit def `Bp2DArray-Bp2DArray`[Input <: Batch]: MathMethods.-.Case.Aux[Layer.Aux[Input, Bp2DArray#Batch],
                                                                             Layer.Aux[Input, Bp2DArray#Batch],
                                                                             Layer.Aux[Input, Bp2DArray#Batch]] = {
    MathMethods.-.at { (leftLayer, rightLayer) =>
      PlusBp2DArray(leftLayer, Negative(rightLayer))
    }
  }

  implicit def `BpDouble-Bp2DArray`[Input <: Batch]: MathMethods.-.Case.Aux[Layer.Aux[Input, BpDouble#Batch],
                                                                            Layer.Aux[Input, Bp2DArray#Batch],
                                                                            Layer.Aux[Input, Bp2DArray#Batch]] = {
    MathMethods.-.at { (leftLayer, rightLayer) =>
      PlusBpDouble(Negative(rightLayer), leftLayer)
    }
  }

  implicit def `Bp2DArray-BpDouble`[Input <: Batch]: MathMethods.-.Case.Aux[Layer.Aux[Input, Bp2DArray#Batch],
                                                                            Layer.Aux[Input, BpDouble#Batch],
                                                                            Layer.Aux[Input, Bp2DArray#Batch]] = {
    MathMethods.-.at { (leftLayer, rightLayer) =>
      PlusBpDouble(leftLayer, BpDouble.Layers.Negative(rightLayer))
    }
  }

  implicit def `Bp2DArray+Bp2DArray`[Input <: Batch]: MathMethods.+.Case.Aux[Layer.Aux[Input, Bp2DArray#Batch],
                                                                             Layer.Aux[Input, Bp2DArray#Batch],
                                                                             Layer.Aux[Input, Bp2DArray#Batch]] = {
    MathMethods.+.at { (leftLayer, rightLayer) =>
      PlusBp2DArray(leftLayer, rightLayer)
    }
  }

  implicit def `Bp2DArray+BpDouble`[Input <: Batch]: MathMethods.+.Case.Aux[Layer.Aux[Input, Bp2DArray#Batch],
                                                                            Layer.Aux[Input, BpDouble#Batch],
                                                                            Layer.Aux[Input, Bp2DArray#Batch]] = {
    MathMethods.+.at { (leftLayer, rightLayer) =>
      PlusBpDouble(leftLayer, rightLayer)
    }
  }

  implicit def `BpDouble+Bp2DArray`[Input <: Batch]: MathMethods.+.Case.Aux[Layer.Aux[Input, BpDouble#Batch],
                                                                            Layer.Aux[Input, Bp2DArray#Batch],
                                                                            Layer.Aux[Input, Bp2DArray#Batch]] = {
    MathMethods.+.at { (leftLayer, rightLayer) =>
      PlusBpDouble(rightLayer, leftLayer)
    }
  }

  implicit def `exp(Bp2DArray)`[Input <: Batch]
    : exp.Case.Aux[Layer.Aux[Input, Bp2DArray#Batch], Layer.Aux[Input, Bp2DArray#Batch]] = {
    exp.at(Exp(_))
  }

  final class TwoDArrayLayerOps[Input <: Batch](differentiable: Layer.Aux[Input, Bp2DArray#Batch]) {

    def dot(right: Layer.Aux[Input, Bp2DArray#Batch]): Layer.Aux[Input, Bp2DArray#Batch] = {
      Dot(differentiable, right)
    }

    def unary_- : Layer.Aux[Input, Bp2DArray#Batch] = {
      Negative(differentiable)
    }

    def toSeq: Layer.Aux[Input, Batch.Aux[Seq[Seq[Eval[Double]]], (Int, (Int, Eval[Double]))]] = {
      ToSeq(differentiable)
    }

  }

  implicit def to2DArrayLayerOps[From, Input <: Batch](from: From)(
      implicit toLayer: ToLayer.OfType[From, Input, Bp2DArray]
  ): TwoDArrayLayerOps[Input] = {
    new TwoDArrayLayerOps(toLayer(from))
  }

  // TODO: Support Array for better performance.
  final class To2DArrayLayerOps[Input <: Batch](
      layerVector: Seq[Seq[Layer.Aux[Input, Batch.Aux[Eval[Double], Eval[Double]]]]]) {
    def toBp2DArray: Layer.Aux[Input, Bp2DArray#Batch] = ToBp2DArray(layerVector)
  }

  implicit def toTo2DArrayLayerOps[Element, Input <: Batch](layerVector: Seq[Seq[Element]])(
      implicit toLayer: ToLayer.OfType[Element, Input, BpDouble]): To2DArrayLayerOps[Input] = {
    new To2DArrayLayerOps(layerVector.view.map(_.view.map(toLayer(_))))
  }

  implicit final class INDArrayOps(ndArray: INDArray) {
    def toWeight[InputData, InputDelta](
        implicit inputType: BackPropagationType[InputData, InputDelta],
        optimizer: Optimizer): Layer.Aux[Batch.Aux[InputData, InputDelta], Bp2DArray#Batch] = {
      Weight(ndArray)
    }
  }

  implicit def ndArrayToLiteral: ToLiteral.Aux[INDArray, Eval[INDArray], Eval[INDArray]] = new ToLiteral[INDArray] {
    override type Data = Eval[INDArray]
    override type Delta = Eval[INDArray]

    override def apply(ndArray: INDArray) = Literal(Eval.now(ndArray))
  }

}
