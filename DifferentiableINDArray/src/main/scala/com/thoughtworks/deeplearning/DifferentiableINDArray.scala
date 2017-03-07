package com.thoughtworks.deeplearning

import cats.implicits._
import cats.{Applicative, Eval, Semigroup, Traverse}
import com.thoughtworks.deeplearning.DifferentiableAny.Trainable
import com.thoughtworks.deeplearning.Layer.{Aux, Batch, CloseableOnce}
import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.DifferentiableINDArray.Layers._
import com.thoughtworks.deeplearning.DifferentiableINDArray.Optimizers._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.DifferentiableInt.Layers.Times
import com.thoughtworks.deeplearning.DifferentiableInt._
import com.thoughtworks.deeplearning.Lift.Layers.Literal
import com.thoughtworks.deeplearning.Layer.Batch.Aux
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import com.thoughtworks.deeplearning.Poly.MathMethods
import com.thoughtworks.deeplearning.Poly.MathMethods.*
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.{IsMax, Sqrt}
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.ops.transforms.Transforms.sign
import org.nd4j.linalg.util.ArrayUtil
import org.nd4s.Implicits._
import shapeless._
import org.nd4j.linalg.ops.transforms.Transforms.sqrt

import language.higherKinds
import language.implicitConversions
import scala.collection.immutable.IndexedSeq

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableINDArray {

  private[deeplearning] trait INDArraySemigroupBatch extends Batch {

    override type Data = INDArray

    override type Delta = INDArray

    protected final def semigroup = new Semigroup[Delta] {
      override def combine(x: Delta, y: Delta): Delta = x + y
    }

  }

  // TODO: Add a test for this method and auto-broadcasting on n-dimension arrays for n > 2
  private[DifferentiableINDArray] def sumAs(outputDeltaValue: INDArray, shape: Array[Int]) = {
    val singleElementDimension = (shape: Seq[Int]).view.zip(outputDeltaValue.shape).zipWithIndex.collect {
      case ((1, originSize), dimension) if originSize > 1 => dimension
    }
    if (singleElementDimension.isEmpty) {
      outputDeltaValue
    } else {
      outputDeltaValue.sum(singleElementDimension.force: _*).reshape(shape: _*)
    }
  }

  private[deeplearning] type INDArrayPlaceholder = Placeholder[INDArray, INDArray]
  private[deeplearning] val INDArrayPlaceholder: INDArrayPlaceholder = implicitly

  object Optimizers {

    trait L1Regularization extends Optimizer {
      protected def l1Regularization: Double

      override def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = {
        super.currentDelta(oldValue, delta + sign(oldValue) * l1Regularization)
      }
    }

    trait L2Regularization extends Optimizer {
      protected def l2Regularization: Double

      override def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = {
        super.currentDelta(oldValue, delta + oldValue * l2Regularization)
      }
    }

    trait Momentum extends Optimizer {
      protected def mu(): Double = 0.9

      private var v: Option[INDArray] = None

      override def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = {
        val vValue: INDArray = v.getOrElse(Nd4j.zeros(delta.shape: _*))
        v = Some(
          super.currentDelta(oldValue, delta) + vValue * mu()
        )
        v.get
      }
    }

    trait NesterovMomentum extends Optimizer {
      protected def mu(): Double = 0.9

      private var v: Option[INDArray] = None

      override def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = {
        val vValue: INDArray = v.getOrElse(Nd4j.zeros(delta.shape: _*))
        val vPre = vValue
        v = Some(
          super.currentDelta(oldValue, delta) + vValue * mu()
        )

        vPre * (-mu()) + v.get * (1 + mu())
      }
    }

    trait Adagrad extends Optimizer {

      protected def eps(): Double = 1e-4

      private var cache: Option[INDArray] = None

      override def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = {
        val cacheValue = cache.getOrElse(Nd4j.zeros(delta.shape: _*))
        cache = Some(cacheValue + delta * delta)
        super.currentDelta(oldValue, delta) / (sqrt(cache.get) + eps)
      }
    }

    trait RMSprop extends Optimizer {

      protected def decayRate(): Double = 0.99

      protected def eps(): Double = 1e-4

      private var cache: Option[INDArray] = None

      override def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = {
        val cacheValue = cache.getOrElse(Nd4j.zeros(delta.shape: _*))
        cache = Some(cacheValue * decayRate + delta * delta * (1 - decayRate))
        super.currentDelta(oldValue, delta) / (sqrt(cache.get) + eps)
      }
    }

    trait Adam extends Optimizer {

      protected def beta1 = 0.9

      protected def beta2 = 0.999

      protected def eps(): Double = 1e-8

      private var m: Option[INDArray] = None

      private var v: Option[INDArray] = None

      private var times: Int = 0

      override def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = {

        val mValue = m.getOrElse(Nd4j.zeros(delta.shape: _*))

        m = Some(
          mValue * beta1 + delta * (1 - beta1)
        )

        val vValue = v.getOrElse(Nd4j.zeros(delta.shape: _*))

        v = Some(
          vValue * beta2 + delta * delta * (1 - beta2)
        )

        times += 1

        val coef1 = 1 - math.pow(beta1, times)

        val coef2 = math.sqrt(1 - math.pow(beta2, times))

        super.currentDelta(oldValue, m.get * (coef2 / coef1)) / (sqrt(v.get) + eps)
      }
    }

    trait Optimizer {

      protected def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = delta

      final def updateNDArray(oldValue: INDArray, delta: INDArray): INDArray = {
        oldValue - currentDelta(oldValue, delta)
      }
    }

    trait LearningRate extends Optimizer {

      protected def currentLearningRate(): Double

      override def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = delta * currentLearningRate()
    }

  }

  import Optimizers._

  object OptimizerFactory {
    implicit def shared(implicit optimizer: Optimizer): OptimizerFactory = new OptimizerFactory {
      override def ndArrayOptimizer(weight: Weight): Optimizer = optimizer
    }
  }

  trait OptimizerFactory {
    def ndArrayOptimizer(weight: Weight): Optimizer
  }

  object Layers {

    private def autoBroadcastShape(shape1: Array[Int], shape2: Array[Int]) = {
      require(shape1.length == shape2.length)
      shape1.zip(shape2).map {
        case (1, bSize) => bSize
        case (aSize, 1) => aSize
        case (aSize, bSize) if aSize == bSize => aSize
      }
    }

    final case class MultiplyINDArray[Input0 <: Batch](
        operand1: Layer.Aux[Input0, INDArrayPlaceholder.Batch],
        operand2: Layer.Aux[Input0, INDArrayPlaceholder.Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = INDArraySemigroupBatch with SemigroupBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with INDArraySemigroupBatch with SemigroupBatch with BinaryBatch {

          val value = {
            val aValue = upstream1.value
            val bValue = upstream2.value
            val newShape = autoBroadcastShape(aValue.shape(), bValue.shape())
            val broadcastA = aValue.broadcast(newShape: _*)
            val broadcastB = bValue.broadcast(newShape: _*)
            val result = broadcastA * broadcastB
            result
          }

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            val a = upstream1.value
            val b = upstream2.value
            upstream1.backward(sumAs(b.broadcast(outputDelta.shape(): _*) * outputDelta, a.shape()))
            upstream2.backward(sumAs(a.broadcast(outputDelta.shape(): _*) * outputDelta, b.shape()))
          }
        }
      }
    }

    final case class MaxDouble[Input0 <: Batch](
        operand1: Layer.Aux[Input0, INDArrayPlaceholder.Batch],
        operand2: Layer.Aux[Input0, DoublePlaceholder.Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = INDArraySemigroupBatch with SemigroupBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with INDArraySemigroupBatch with SemigroupBatch with BinaryBatch {

          val value = Transforms.max(upstream1.value, upstream2.value)

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            val a = upstream1.value
            val b = upstream2.value
            upstream1.backward((a gt b) * outputDelta)
            upstream2.backward(((a lt b) * outputDelta).sumT)
          }
        }
      }
    }

    final case class PlusINDArray[Input0 <: Batch](
        operand1: Layer.Aux[Input0, INDArrayPlaceholder.Batch],
        operand2: Layer.Aux[Input0, INDArrayPlaceholder.Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = INDArraySemigroupBatch with SemigroupBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with INDArraySemigroupBatch with SemigroupBatch with BinaryBatch {

          val value = {
            val aValue = upstream1.value
            val bValue = upstream2.value
            val newShape = autoBroadcastShape(aValue.shape(), bValue.shape())
            aValue.broadcast(newShape: _*) + bValue.broadcast(newShape: _*)
          }

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            upstream1.backward(sumAs(outputDelta, upstream1.value.shape()))
            upstream2.backward(sumAs(outputDelta, upstream2.value.shape()))
          }
        }
      }
    }

    object ToSeq {
      private[ToSeq] trait Seq2DBatch extends Batch {
        override type Data = Seq[Seq[Double]]
        override type Delta = (Int, (Int, Double))
      }
    }

    /**
      * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
      */
    final case class ToSeq[Input0 <: Batch](operand: Layer.Aux[Input0, INDArrayPlaceholder.Batch])
        extends BufferedLayer.Unary {
      import ToSeq._
      type BufferedBatch =
        UnaryBatch with Seq2DBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with UnaryBatch with Seq2DBatch {

          private def zeroDelta = Nd4j.zeros(upstream.value.shape: _*)

          @volatile
          var upstreamDelta = zeroDelta

          override protected def flush(): Unit = {
            upstream.backward(synchronized {
              val oldDelta = upstreamDelta
              upstreamDelta = zeroDelta
              oldDelta
            })
          }

          override protected def forceBackward(delta: Delta): Unit = {
            synchronized {
              val (i, (j, value)) = delta
              // Cannot use += because of https://issues.scala-lang.org/browse/SI-10021
              upstreamDelta(i, j) = upstreamDelta(i, j) + value
            }
          }

          override val value: Data = {
            val ndarray: INDArray = upstream.value
            val doubleArray: Seq[Double] = ndarray.data.asDouble()
            doubleArray.grouped(ndarray.columns).toSeq
//            for (i <- (0 until ndarray.rows)) yield {
//              doubleArray.view(i * ndarray.columns, (i + 1) * ndarray.columns)
//            }
          }
        }
      }
    }

    object Weight {
      def apply(value: INDArray)(implicit optimizerFactory: OptimizerFactory) = new Weight(value) {
        override protected val optimizer = optimizerFactory.ndArrayOptimizer(this)
      }
    }

    abstract case class Weight(var value: INDArray) extends Layer with INDArraySemigroupBatch {

      protected def optimizer: Optimizer

      override type Input = Batch
      override type Output = Batch.Aux[Data, Delta]

      override final val isTrainable = true

      override final def addReference() = this

      override final def forward(any: Input) = this

      override final protected def forceBackward(delta: Delta): Unit = {
        synchronized {
          value = optimizer.updateNDArray(value, delta)
        }
      }

      override final def close(): Unit = {}

    }

    final case class ToINDArray[Input0 <: Batch](operands: Seq[Seq[Layer.Aux[Input0, Batch.Aux[Double, Double]]]])
        extends Layer {

      type Input = Input0

      final class Output private[ToINDArray] (upstreams: Seq[Seq[Batch.Aux[Double, Double]]])
          extends INDArraySemigroupBatch
          with CloseableOnce {

        override val isTrainable = upstreams.exists(_.exists(_.isTrainable))

        override protected def forceBackward(delta: INDArray): Unit = {
          for ((row, i) <- upstreams.view.zipWithIndex; (upstream, j) <- row.view.zipWithIndex) {
            upstream.backward(delta(i, j))
          }

        }

        override val value = {
          upstreams.map(_.map(_.value)).toNDArray
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

    final case class Sum[Input0 <: Batch](operand: Layer.Aux[Input0, INDArrayPlaceholder.Batch], dimensions: Seq[Int])
        extends BufferedLayer.Unary {
      type BufferedBatch = INDArraySemigroupBatch with SemigroupBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with INDArraySemigroupBatch with SemigroupBatch with UnaryBatch {

          val value = upstream.value.sum(dimensions: _*)

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            val a = upstream.value
            upstream.backward(outputDelta.broadcast(a.shape: _*))
          }
        }
      }
    }

    final case class ReduceSum[Input0 <: Batch](operand: Layer.Aux[Input0, INDArrayPlaceholder.Batch])
        extends BufferedLayer.Unary {
      type BufferedBatch = DoubleMonoidBatch with MonoidBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with DoubleMonoidBatch with MonoidBatch with UnaryBatch {

          val value = (upstream.value: INDArray).sumT

          override protected def rawBackward(outputDelta: Double): Unit = {
            upstream.backward(Nd4j.valueArrayOf(upstream.value.shape(), outputDelta))
          }
        }
      }
    }

    final case class Reciprocal[Input0 <: Batch](operand: Layer.Aux[Input0, INDArrayPlaceholder.Batch])
        extends BufferedLayer.Unary {
      type BufferedBatch = INDArraySemigroupBatch with SemigroupBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with INDArraySemigroupBatch with SemigroupBatch with UnaryBatch {

          val value = upstream.value rdiv 1.0

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            val upstreamValue: INDArray = upstream.value
            upstream.backward(-outputDelta / (upstreamValue * upstreamValue))
          }
        }
      }
    }

    final case class PlusDouble[Input0 <: Batch](
        operand1: Layer.Aux[Input0, INDArrayPlaceholder.Batch],
        operand2: Layer.Aux[Input0, DoublePlaceholder.Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = INDArraySemigroupBatch with SemigroupBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with INDArraySemigroupBatch with SemigroupBatch with BinaryBatch {
          val value = (upstream1.value: INDArray) + upstream2.value

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            upstream1.backward(outputDelta)
            upstream2.backward(outputDelta.sumT)
          }
        }
      }
    }

    final case class Negative[Input0 <: Batch](operand: Layer.Aux[Input0, INDArrayPlaceholder.Batch])
        extends BufferedLayer.Unary {
      type BufferedBatch = INDArraySemigroupBatch with SemigroupBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with INDArraySemigroupBatch with SemigroupBatch with UnaryBatch {

          val value = -upstream.value

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            upstream.backward(-outputDelta)
          }
        }
      }
    }

    final case class Exp[Input0 <: Batch](operand: Layer.Aux[Input0, INDArrayPlaceholder.Batch])
        extends BufferedLayer.Unary {
      type BufferedBatch = INDArraySemigroupBatch with SemigroupBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with INDArraySemigroupBatch with SemigroupBatch with UnaryBatch {
          val value = Transforms.exp(upstream.value)

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            upstream.backward(value * outputDelta)
          }
        }
      }
    }

    final case class Abs[Input0 <: Batch](operand: Layer.Aux[Input0, INDArrayPlaceholder.Batch])
        extends BufferedLayer.Unary {
      type BufferedBatch = INDArraySemigroupBatch with SemigroupBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with INDArraySemigroupBatch with SemigroupBatch with UnaryBatch {

          val value = Transforms.abs(upstream.value)

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            upstream.backward(outputDelta * Transforms.sign(upstream.value))
          }
        }
      }
    }

    final case class Log[Input0 <: Batch](operand: Layer.Aux[Input0, INDArrayPlaceholder.Batch])
        extends BufferedLayer.Unary {
      type BufferedBatch = INDArraySemigroupBatch with SemigroupBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with INDArraySemigroupBatch with SemigroupBatch with UnaryBatch {

          val value = Transforms.log(upstream.value)

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            upstream.backward(outputDelta / upstream.value)
          }
        }
      }
    }

    final case class MultiplyDouble[Input0 <: Batch](
        operand1: Layer.Aux[Input0, INDArrayPlaceholder.Batch],
        operand2: Layer.Aux[Input0, DoublePlaceholder.Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = INDArraySemigroupBatch with SemigroupBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with INDArraySemigroupBatch with SemigroupBatch with BinaryBatch {

          val value = (upstream1.value: INDArray) * upstream2.value

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            val a: INDArray = upstream1.value
            val b = upstream2.value

            val aDelta = outputDelta * b
            upstream1.backward(aDelta)
            val bDelta = (a * outputDelta).sumT
            upstream2.backward(bDelta)
          }
        }
      }
    }

    // TODO: Support n-dimension array when n > 2
    // See https://www.tensorflow.org/api_docs/python/math_ops/matrix_math_functions#matmul for the behavior
    final case class Dot[Input0 <: Batch](
        operand1: Layer.Aux[Input0, INDArrayPlaceholder.Batch],
        operand2: Layer.Aux[Input0, INDArrayPlaceholder.Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = INDArraySemigroupBatch with SemigroupBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with INDArraySemigroupBatch with SemigroupBatch with BinaryBatch {

          override val value = (upstream1.value: INDArray) dot upstream2.value

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            val b: INDArray = upstream2.value
            upstream1.backward(outputDelta dot b.T)
            val a: INDArray = upstream1.value
            upstream2.backward(a.T.dot(outputDelta))
          }
        }

      }
    }

    final case class Im2col[Input0 <: Batch](
        operand: Layer.Aux[Input0, INDArrayPlaceholder.Batch],
        kernel: Array[Int],
        stride: Array[Int],
        padding: Array[Int]
    ) extends BufferedLayer.Unary {
      type BufferedBatch = INDArraySemigroupBatch with SemigroupBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with INDArraySemigroupBatch with SemigroupBatch with UnaryBatch {

          private val upstreamShape = {
            upstream.value.shape()
          }

          val value = Convolution.im2col(upstream.value, kernel, stride, padding)

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            upstream.backward(Convolution.col2im(outputDelta, stride, padding, upstreamShape(2), upstreamShape(3)))
          }
        }
      }
    }

    final case class Reshape[Input0 <: Batch](
        override val operand1: Layer.Aux[Input0, INDArrayPlaceholder.Batch],
        override val operand2: Layer.Aux[Input0, Batch.Aux[Seq[Int], (Int, Float)]])
        extends BufferedLayer.Binary {
      override type BufferedBatch = INDArraySemigroupBatch with SemigroupBatch with BinaryBatch

      override type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with INDArraySemigroupBatch with SemigroupBatch with BinaryBatch {

          private val upstreamShape = {
            upstream1.value.shape
          }

          override val value = upstream1.value.reshape(upstream2.value: _*)

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            upstream1.backward(outputDelta.reshape(upstreamShape: _*))
          }
        }
      }
    }

    final case class Permute[Input0 <: Batch](
        override val operand1: Layer.Aux[Input0, INDArrayPlaceholder.Batch],
        override val operand2: Layer.Aux[Input0, Batch.Aux[Seq[Int], (Int, Float)]])
        extends BufferedLayer.Binary {
      override type BufferedBatch = INDArraySemigroupBatch with SemigroupBatch with BinaryBatch

      override type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with INDArraySemigroupBatch with SemigroupBatch with BinaryBatch {

          private val upstreamShape: Seq[Int] = {
            upstream1.value.shape()
          }

          override val value = upstream1.value.permute(upstream2.value: _*)

          override protected def rawBackward(outputDelta: INDArray): Unit = {

            val indexSeq: IndexedSeq[Int] =
              upstreamShape.indices
                .map(
                  index => upstream2.value.toSeq.indexOf(index)
                )

            upstream1.backward(
              outputDelta.permute(indexSeq: _*)
            )
          }
        }
      }
    }

    final case class MaxPool[Input0 <: Batch](override val operand: Layer.Aux[Input0, INDArrayPlaceholder.Batch],
                                              dimensions: Int*)
        extends BufferedLayer.Unary {
      override type BufferedBatch = INDArraySemigroupBatch with SemigroupBatch with UnaryBatch

      override type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with INDArraySemigroupBatch with SemigroupBatch with UnaryBatch {

          if (dimensions.length > 2) {
            throw new UnsupportedOperationException("dimentions's length must <2")
          }

          private val upstreamShape = {
            upstream.value.shape()
          }

          private val isReshape = {
            dimensions.length > 1
          }

          private val lastShapeSize = {
            upstreamShape.reverse
              .take(dimensions.length)
              .product
          }

          private val afterMaxPoolShape = {
            upstreamShape.take(upstreamShape.length - dimensions.length)
          }

          private val reshapeTo = {
            afterMaxPoolShape :+ lastShapeSize
          }

          override val value =
            if (isReshape)
              upstream.value
                .reshape(reshapeTo: _*)
                .max(dimensions(0))
            else upstream.value.max(dimensions(0))

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            val a = upstream.value
            val upStreamDup = a.dup()
            val rows = ArrayUtil.prod(a.length())
            val isMax: INDArray =
              if (isReshape) {
                Nd4j.getExecutioner
                  .execAndReturn(new IsMax(upStreamDup.reshape(reshapeTo: _*), dimensions(0)))
              } else {
                Nd4j.getExecutioner
                  .execAndReturn(new IsMax(upStreamDup, dimensions(0)))
              }

            val outputDelta1d = a
//              (if (isReshape) {
//                 outputDelta
//                   .repeat(-1, Seq(upstreamShape(dimensions(1))): _*)
//                   .permute(1, 0, 3, 2)
//                   .repeat(-1, Seq(upstreamShape(dimensions(0))): _*)
//                   .permute(1, 0, 3, 2)
//              } else {
//                 outputDelta.repeat(dimensions(0), Seq(lastShapeSize): _*)
//              }).reshape('c', rows, 1)
            upstream.backward(
              isMax
                .reshape('c', rows, 1)
                .muliColumnVector(outputDelta1d)
                .reshape(upstreamShape: _*)
            )
          }
        }
      }
    }

    final case class Shape[Input0 <: Batch](operand: Layer.Aux[Input0, INDArrayPlaceholder.Batch]) extends Layer {
      override def forward(input: Input0): Output = {
        val upstream = operand.forward(input)
        try {
          val upstreamShape = upstream.value.shape()
          Literal[Seq[Int]](upstreamShape)
        } finally {
          upstream.close()
        }
      }
      override type Input = Input0
      override type Output = Literal[Seq[Int]]
    }

  }

  import Layers._

  implicit def `max(INDArray,Double)`[Left, Right, Input <: Batch]
    : max.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Batch],
                   Layer.Aux[Input, DoublePlaceholder.Batch],
                   Layer.Aux[Input, INDArrayPlaceholder.Batch]] =
    max.at(MaxDouble(_, _))

  implicit def `INDArray/INDArray`[Input <: Batch]
    : MathMethods./.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch]] = {
    MathMethods./.at { (leftLayer, rightLayer) =>
      MultiplyINDArray(leftLayer, Reciprocal(rightLayer))
    }
  }

  implicit def `Double/INDArray`[Input <: Batch]
    : MathMethods./.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch]] = {
    MathMethods./.at { (leftLayer, rightLayer) =>
      MultiplyDouble(Reciprocal(rightLayer), leftLayer)
    }
  }

  implicit def `INDArray/Double`[Input <: Batch]
    : MathMethods./.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Batch],
                             Layer.Aux[Input, DoublePlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch]] = {
    MathMethods./.at { (leftLayer, rightLayer) =>
      MultiplyDouble(leftLayer, DifferentiableDouble.Layers.Reciprocal(rightLayer))
    }
  }

  implicit def `INDArray*INDArray`[Input <: Batch]
    : MathMethods.*.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch]] = {
    MathMethods.*.at { (leftLayer, rightLayer) =>
      MultiplyINDArray(leftLayer, rightLayer)
    }
  }

  implicit def `INDArray*Double`[Input <: Batch]
    : MathMethods.*.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Batch],
                             Layer.Aux[Input, DoublePlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch]] = {
    MathMethods.*.at { (leftLayer, rightLayer) =>
      MultiplyDouble(leftLayer, rightLayer)
    }
  }

  implicit def `Double*INDArray`[Input <: Batch]
    : MathMethods.*.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch]] = {
    MathMethods.*.at { (leftLayer, rightLayer) =>
      MultiplyDouble(rightLayer, leftLayer)
    }
  }

  implicit def `INDArray-INDArray`[Input <: Batch]
    : MathMethods.-.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch]] = {
    MathMethods.-.at { (leftLayer, rightLayer) =>
      PlusINDArray(leftLayer, Negative(rightLayer))
    }
  }

  implicit def `Double-INDArray`[Input <: Batch]
    : MathMethods.-.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch]] = {
    MathMethods.-.at { (leftLayer, rightLayer) =>
      PlusDouble(Negative(rightLayer), leftLayer)
    }
  }

  implicit def `INDArray-Double`[Input <: Batch]
    : MathMethods.-.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Batch],
                             Layer.Aux[Input, DoublePlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch]] = {
    MathMethods.-.at { (leftLayer, rightLayer) =>
      PlusDouble(leftLayer, DifferentiableDouble.Layers.Negative(rightLayer))
    }
  }

  implicit def `INDArray+INDArray`[Input <: Batch]
    : MathMethods.+.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch]] = {
    MathMethods.+.at { (leftLayer, rightLayer) =>
      PlusINDArray(leftLayer, rightLayer)
    }
  }

  implicit def `INDArray+Double`[Input <: Batch]
    : MathMethods.+.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Batch],
                             Layer.Aux[Input, DoublePlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch]] = {
    MathMethods.+.at { (leftLayer, rightLayer) =>
      PlusDouble(leftLayer, rightLayer)
    }
  }

  implicit def `Double+INDArray`[Input <: Batch]
    : MathMethods.+.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch],
                             Layer.Aux[Input, INDArrayPlaceholder.Batch]] = {
    MathMethods.+.at { (leftLayer, rightLayer) =>
      PlusDouble(rightLayer, leftLayer)
    }
  }

  implicit def `exp(INDArray)`[Input <: Batch]
    : exp.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Batch], Layer.Aux[Input, INDArrayPlaceholder.Batch]] = {
    exp.at(Exp(_))
  }

  implicit def `log(INDArray)`[Input <: Batch]
    : log.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Batch], Layer.Aux[Input, INDArrayPlaceholder.Batch]] = {
    log.at(Log(_))
  }

  implicit def `abs(INDArray)`[Input <: Batch]
    : abs.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Batch], Layer.Aux[Input, INDArrayPlaceholder.Batch]] = {
    abs.at(Abs(_))
  }

  final class INDArrayLayerOps[Input <: Batch](operand: Layer.Aux[Input, INDArrayPlaceholder.Batch]) {

    // TODO: Considering if rename this method to `matmul`
    def dot(right: Layer.Aux[Input, INDArrayPlaceholder.Batch]): Layer.Aux[Input, INDArrayPlaceholder.Batch] = {
      Dot(operand, right)
    }

    def im2col(kernel: Array[Int],
               stride: Array[Int],
               padding: Array[Int]): Layer.Aux[Input, INDArrayPlaceholder.Batch] = {
      Im2col(operand, kernel, stride, padding)
    }

    def dynamicReshape(
        newShape: Layer.Aux[Input, Batch.Aux[Seq[Int], (Int, Float)]]): Layer.Aux[Input, INDArrayPlaceholder.Batch] = {
      Reshape(operand, newShape)
    }

    /**
      * @usecase def reshape(newShape: Layer.Aux[Input, Batch.Aux[Int, Float]]*): Layer.Aux[Input, INDArrayPlaceholder.Batch] = ???
      * @usecase def reshape(newShape: Int*): Layer.Aux[Input, INDArrayPlaceholder.Batch] = ???
      */
    def reshape[Element](newShape: Element*)(
        implicit toLayer: ToLayer.Aux[Element, Input, Int, Float]): Layer.Aux[Input, INDArrayPlaceholder.Batch] = {
      Reshape(operand, DifferentiableSeq.Layers.ToSeq(newShape.map(toLayer.apply(_))))
    }

    /**
      * @usecase def permute(newShape: Layer.Aux[Input, Batch.Aux[Int, Float]]*): Layer.Aux[Input, INDArrayPlaceholder.Batch] = ???
      * @usecase def permute(newShape: Int*): Layer.Aux[Input, INDArrayPlaceholder.Batch] = ???
      */
    def permute[Element](newShape: Element*)(
        implicit toLayer: ToLayer.Aux[Element, Input, Int, Float]): Layer.Aux[Input, INDArrayPlaceholder.Batch] = {
      Permute(operand, DifferentiableSeq.Layers.ToSeq(newShape.map(toLayer.apply(_))))
    }

    private def toArray(tuple2: (Int, Int)): Array[Int] = {
      val (one, two) = tuple2
      Array(one, two)
    }

    /**
      * calculate the convolution
      * @param weight 4 dimensions weight
      * @param bias 1 dimension bias
      * @param kernel the kernel width and height
      * @param stride the stride width and height
      * @param padding the padding width and height
      * @return convolution result
      */
    def convn(weight: Layer.Aux[Input, INDArrayPlaceholder.Batch],
              bias: Layer.Aux[Input, INDArrayPlaceholder.Batch],
              kernel: (Int, Int),
              stride: (Int, Int),
              padding: (Int, Int)): Layer.Aux[Input, INDArrayPlaceholder.Batch] = {
      val shapeOfOperand = Shape(operand)
      val count = DifferentiableSeq.Layers.Get(shapeOfOperand, 0)
      val depth = DifferentiableSeq.Layers.Get(shapeOfOperand, 1)
      val y_axis = DifferentiableSeq.Layers.Get(shapeOfOperand, 2)
      val x_axis = DifferentiableSeq.Layers.Get(shapeOfOperand, 3)
      val kernelNumber = DifferentiableSeq.Layers.Get(Shape(weight), 0)

      //input
      //  .im2col(Array(KernelSize, KernelSize),
      //    Array(Stride, Stride),
      //    Array(Padding, Padding))
      val col: Layer.Aux[Input, Batch.Aux[INDArray, INDArray]] =
        Im2col(operand, toArray(kernel), toArray(stride), toArray(padding))

      //permute(0, 4, 5, 1, 2, 3)
      val permutedCol: Layer.Aux[Input, Batch.Aux[INDArray, INDArray]] = Permute(col, Literal(Seq(0, 4, 5, 1, 2, 3)))

      val depthKernelKernel: Layer.Aux[Input, Batch.Aux[Int, Float]] =
        Times(
          Times(depth, Literal(kernel._1)),
          Literal(kernel._2)
        )

      val countXaxisYaxis: Layer.Aux[Input, Batch.Aux[Int, Float]] = Times(Times(count, y_axis), x_axis)

      val aSeq: Seq[Layer.Aux[Input, Batch.Aux[Int, Float]]] = Seq(countXaxisYaxis, depthKernelKernel)

      val reshapeOperandTo: Layer.Aux[Input, Batch.Aux[Seq[Int], (Int, Float)]] = DifferentiableSeq.Layers.ToSeq(aSeq)

      //reshape(imageCount * inputSizeY * inputSizeX,(depth * KernelSize * KernelSize).toLayer)
      val operandCol2d = Reshape(permutedCol, reshapeOperandTo)

      val bSeq: Seq[Layer.Aux[Input, Batch.Aux[Int, Float]]] = Seq(kernelNumber, depthKernelKernel)

      val reshapeWeightTo: Layer.Aux[Input, Batch.Aux[Seq[Int], (Int, Float)]] = DifferentiableSeq.Layers.ToSeq(bSeq)

      //weight.reshape(kernelNumber, KernelSize * KernelSize * depth)
      val reshapedWeight = Reshape(weight, reshapeWeightTo)

      //permute(1, 0)
      val permutedWeight = Permute(reshapedWeight, Literal(Seq(1, 0)))

      val dotResult = Dot(operandCol2d, permutedWeight)

      val plusResult = PlusINDArray(dotResult, bias)

      val SeqOfCountYaxisXaxisKernelNumber: Seq[Layer.Aux[Input, Batch.Aux[Int, Float]]] =
        Seq(count, y_axis, x_axis, kernelNumber)

      val reshapeResultTo: Layer.Aux[Input, Batch.Aux[Seq[Int], (Int, Float)]] =
        DifferentiableSeq.Layers.ToSeq(SeqOfCountYaxisXaxisKernelNumber)

      //reshape(imageCount, inputSizeY, inputSizeX, kernelNumber.toLayer)
      val reshapedResult = Reshape(plusResult, reshapeResultTo)

      val permuteResultTo = Literal(Seq(0, 3, 1, 2))

      //permute(0, 3, 1, 2)
      Permute(reshapedResult, permuteResultTo)
    }

    def maxPool(dimensions: Int*): Layer.Aux[Input, INDArrayPlaceholder.Batch] = {
      MaxPool(operand, dimensions: _*)
    }

    def shape: Layer.Aux[Input, Batch.Aux[Seq[Int], (Int, Float)]] = {
      Shape(operand)
    }

    def unary_- : Layer.Aux[Input, INDArrayPlaceholder.Batch] = {
      Negative(operand)
    }

    def toSeq: Layer.Aux[Input, Batch.Aux[Seq[Seq[Double]], (Int, (Int, Double))]] = {
      ToSeq(operand)
    }

    def sum: Layer.Aux[Input, DoublePlaceholder.Batch] = {
      ReduceSum(operand)
    }

    def sum(dimensions: Int*): Layer.Aux[Input, INDArrayPlaceholder.Batch] = {
      Sum(operand, dimensions)
    }

  }

  implicit def toINDArrayLayerOps[From, Input <: Batch, OutputData, OutputDelta](from: From)(
      implicit toLayer: ToLayer.Aux[From, Input, OutputData, OutputDelta],
      constrait: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] <:< Layer.Aux[Input,
                                                                                    Batch.Aux[INDArray, INDArray]]
  ): INDArrayLayerOps[Input] = {
    new INDArrayLayerOps(constrait(toLayer(from)))
  }

  // TODO: Support Array for better performance.
  final class ToINDArrayLayerOps[Input <: Batch](layerVector: Seq[Seq[Layer.Aux[Input, Batch.Aux[Double, Double]]]]) {
    def toINDArray: Layer.Aux[Input, INDArrayPlaceholder.Batch] = ToINDArray(layerVector)
  }

  implicit def toToINDArrayLayerOps[Element, Input <: Batch](layerVector: Seq[Seq[Element]])(
      implicit toLayer: ToLayer.OfPlaceholder[Element, Input, DoublePlaceholder]): ToINDArrayLayerOps[Input] = {
    new ToINDArrayLayerOps(layerVector.map(_.map(toLayer(_))))
  }

  implicit final class INDArrayOps(ndArray: INDArray) {
    def toWeight[InputData, InputDelta](
        implicit inputType: Placeholder[InputData, InputDelta],
        optimizerFactory: OptimizerFactory): Layer.Aux[Batch.Aux[InputData, InputDelta], INDArrayPlaceholder.Batch] = {
      Weight(ndArray)
    }
  }

  implicit def liftINDArray: Lift.Aux[INDArray, INDArray, INDArray] = Lift.fromData

  implicit def indArrayTrainable: Trainable[INDArray, INDArray] = new Trainable[INDArray, INDArray] {
    override def apply(data: INDArray): INDArray = Nd4j.ones(data.shape(): _*)
  }

}
