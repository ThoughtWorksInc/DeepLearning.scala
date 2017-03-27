package com.thoughtworks.deeplearning

import cats.implicits._
import cats.{Applicative, Eval, Semigroup, Traverse}
import com.thoughtworks.deeplearning.DifferentiableAny.Trainable
import com.thoughtworks.deeplearning.Layer.{Aux, Tape, CloseableOnce}
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.DifferentiableINDArray.Layers._
import com.thoughtworks.deeplearning.DifferentiableINDArray.Optimizers._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.DifferentiableInt.Layers.Times
import com.thoughtworks.deeplearning.DifferentiableInt._
import com.thoughtworks.deeplearning.Symbolic.Layers.Literal
import com.thoughtworks.deeplearning.Layer.Tape.Aux
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
  * A namespace of common operators for [[org.nd4j.linalg.api.ndarray.INDArray INDArray]] layers.
  *
  * After importing `DifferentiableINDArray._`,
  *
  * You will able to use [[Poly.MathFunctions MathFunctions]],like
  *  - [[DifferentiableINDArray.log(INDArray) log]]
  *
  * You will able to use [[Poly.MathMethods MathMethods]],like
  *  - [[DifferentiableINDArray.INDArray+Double +]]
  *
  * You will able to use [[DifferentiableINDArray.INDArrayLayerOps INDArrayLayerOps]],like
  *  - [[DifferentiableINDArray.INDArrayLayerOps.im2col im2col]]
  *
  * You will able to use some methods like [[DifferentiableINDArray.conv2d conv2d]]
  *
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableINDArray {

  private[deeplearning] trait INDArraySemigroupTape extends Tape {

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

  /**
    * A namespaces contains [[Optimizer]]s for [[org.nd4j.linalg.api.ndarray.INDArray INDArray]].
    *
    * @see [[DifferentiableINDArray.Layers.Weight Weight]]
    *
    */
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

  /**
    * If you write something like this:
    * {{{
    * implicit def optimizer: Optimizer = new LearningRate {
    *   def currentLearningRate() = 0.001
    * }
    * }}}
    * the learningRate will shared with all layers. If a [[DifferentiableINDArray.Optimizers.Optimizer Optimizer]] has state, then learningRate can NOT been shared.
    *
    */
  object OptimizerFactory {
    implicit def shared(implicit optimizer: Optimizer): OptimizerFactory = new OptimizerFactory {
      override def ndArrayOptimizer(weight: Weight): Optimizer = optimizer
    }
  }

  /**
    * If a [[DifferentiableINDArray.Optimizers.Optimizer Optimizer]] has state, then learningRate can NOT been shared, such as [[DifferentiableINDArray.Optimizers.Adam Adam]], so you will need:
    *
    * @example{{{
    * implicit val optimizerFactory = new DifferentiableINDArray.OptimizerFactory {
    *   override def ndArrayOptimizer(weight: Weight): Optimizer = {
    *     new LearningRate with L2Regularization with Adam {
    *
    *       var learningRate = 0.00003
    *
    *       override protected def l2Regularization: Double = 0.00003
    *
    *       override protected def currentLearningRate(): Double = {
    *       learningRate * 0.75
    *       learningRate
    *      }
    *    }
    *  }
    * }
    * }}}
    */
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

    final case class MultiplyINDArray[Input0 <: Tape](
        operand1: Layer.Aux[Input0, INDArrayPlaceholder.Tape],
        operand2: Layer.Aux[Input0, INDArrayPlaceholder.Tape]
    ) extends CumulativeLayer.Binary {

      type CumulativeTape = INDArraySemigroupTape with SemigroupTape with BinaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override final val input = input0
        } with INDArraySemigroupTape with SemigroupTape with BinaryTape {

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

    final case class MaxDouble[Input0 <: Tape](
        operand1: Layer.Aux[Input0, INDArrayPlaceholder.Tape],
        operand2: Layer.Aux[Input0, DoublePlaceholder.Tape]
    ) extends CumulativeLayer.Binary {

      type CumulativeTape = INDArraySemigroupTape with SemigroupTape with BinaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override final val input = input0
        } with INDArraySemigroupTape with SemigroupTape with BinaryTape {

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

    final case class PlusINDArray[Input0 <: Tape](
        operand1: Layer.Aux[Input0, INDArrayPlaceholder.Tape],
        operand2: Layer.Aux[Input0, INDArrayPlaceholder.Tape]
    ) extends CumulativeLayer.Binary {

      type CumulativeTape = INDArraySemigroupTape with SemigroupTape with BinaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override final val input = input0
        } with INDArraySemigroupTape with SemigroupTape with BinaryTape {

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
      private[ToSeq] trait Seq2DTape extends Tape {
        override type Data = Seq[Seq[Double]]
        override type Delta = (Int, (Int, Double))
      }
    }

    /**
      * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
      */
    final case class ToSeq[Input0 <: Tape](operand: Layer.Aux[Input0, INDArrayPlaceholder.Tape])
        extends CumulativeLayer.Unary {
      import ToSeq._
      type CumulativeTape =
        UnaryTape with Seq2DTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override val input = input0
        } with UnaryTape with Seq2DTape {

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

    abstract case class Weight(var value: INDArray) extends Layer with INDArraySemigroupTape {

      protected def optimizer: Optimizer

      override type Input = Tape
      override type Output = Tape.Aux[Data, Delta]

      override final def isTrainable = true

      override final def duplicate() = this

      override final def forward(any: Input) = this

      override final protected def forceBackward(delta: Delta): Unit = {
        synchronized {
          value = optimizer.updateNDArray(value, delta)
        }
      }

      override final def close(): Unit = {}

    }

    final case class ToINDArray[Input0 <: Tape](operands: Seq[Seq[Layer.Aux[Input0, Tape.Aux[Double, Double]]]])
        extends Layer {

      type Input = Input0

      final class Output private[ToINDArray] (upstreams: Seq[Seq[Tape.Aux[Double, Double]]])
          extends INDArraySemigroupTape
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

        override def duplicate(): Output = {
          new Output(upstreams.map(_.map(_.duplicate())))
        }
      }

      override def forward(input: Input) = {
        new Output(operands.map(_.map(_.forward(input))))
      }
    }

    final case class Sum[Input0 <: Tape](operand: Layer.Aux[Input0, INDArrayPlaceholder.Tape], dimensions: Seq[Int])
        extends CumulativeLayer.Unary {
      type CumulativeTape = INDArraySemigroupTape with SemigroupTape with UnaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override val input = input0
        } with INDArraySemigroupTape with SemigroupTape with UnaryTape {

          val value = upstream.value.sum(dimensions: _*)

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            val a = upstream.value
            upstream.backward(outputDelta.broadcast(a.shape: _*))
          }
        }
      }
    }

    final case class ReduceSum[Input0 <: Tape](operand: Layer.Aux[Input0, INDArrayPlaceholder.Tape])
        extends CumulativeLayer.Unary {
      type CumulativeTape = DoubleMonoidTape with MonoidTape with UnaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override val input = input0
        } with DoubleMonoidTape with MonoidTape with UnaryTape {

          val value = (upstream.value: INDArray).sumT

          override protected def rawBackward(outputDelta: Double): Unit = {
            upstream.backward(Nd4j.valueArrayOf(upstream.value.shape(), outputDelta))
          }
        }
      }
    }

    final case class ReduceMean[Input0 <: Tape](operand: Layer.Aux[Input0, INDArrayPlaceholder.Tape])
        extends CumulativeLayer.Unary {
      type CumulativeTape = DoubleMonoidTape with MonoidTape with UnaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override val input = input0
        } with DoubleMonoidTape with MonoidTape with UnaryTape {

          private val upstreamShape = upstream.value.shape()

          val value = (upstream.value: INDArray).sumT / ArrayUtil.prod(upstreamShape: _*)

          override protected def rawBackward(outputDelta: Double): Unit = {
            upstream.backward(Nd4j.valueArrayOf(upstreamShape, outputDelta))
          }
        }
      }
    }

    final case class Reciprocal[Input0 <: Tape](operand: Layer.Aux[Input0, INDArrayPlaceholder.Tape])
        extends CumulativeLayer.Unary {
      type CumulativeTape = INDArraySemigroupTape with SemigroupTape with UnaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override val input = input0
        } with INDArraySemigroupTape with SemigroupTape with UnaryTape {

          val value = upstream.value rdiv 1.0

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            val upstreamValue: INDArray = upstream.value
            upstream.backward(-outputDelta / (upstreamValue * upstreamValue))
          }
        }
      }
    }

    final case class PlusDouble[Input0 <: Tape](
        operand1: Layer.Aux[Input0, INDArrayPlaceholder.Tape],
        operand2: Layer.Aux[Input0, DoublePlaceholder.Tape]
    ) extends CumulativeLayer.Binary {

      type CumulativeTape = INDArraySemigroupTape with SemigroupTape with BinaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override final val input = input0
        } with INDArraySemigroupTape with SemigroupTape with BinaryTape {
          val value = (upstream1.value: INDArray) + upstream2.value

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            upstream1.backward(outputDelta)
            upstream2.backward(outputDelta.sumT)
          }
        }
      }
    }

    final case class Negative[Input0 <: Tape](operand: Layer.Aux[Input0, INDArrayPlaceholder.Tape])
        extends CumulativeLayer.Unary {
      type CumulativeTape = INDArraySemigroupTape with SemigroupTape with UnaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override val input = input0
        } with INDArraySemigroupTape with SemigroupTape with UnaryTape {

          val value = -upstream.value

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            upstream.backward(-outputDelta)
          }
        }
      }
    }

    final case class Exp[Input0 <: Tape](operand: Layer.Aux[Input0, INDArrayPlaceholder.Tape])
        extends CumulativeLayer.Unary {
      type CumulativeTape = INDArraySemigroupTape with SemigroupTape with UnaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override val input = input0
        } with INDArraySemigroupTape with SemigroupTape with UnaryTape {
          val value = Transforms.exp(upstream.value)

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            upstream.backward(value * outputDelta)
          }
        }
      }
    }

    final case class Abs[Input0 <: Tape](operand: Layer.Aux[Input0, INDArrayPlaceholder.Tape])
        extends CumulativeLayer.Unary {
      type CumulativeTape = INDArraySemigroupTape with SemigroupTape with UnaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override val input = input0
        } with INDArraySemigroupTape with SemigroupTape with UnaryTape {

          val value = Transforms.abs(upstream.value)

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            upstream.backward(outputDelta * Transforms.sign(upstream.value))
          }
        }
      }
    }

    final case class Log[Input0 <: Tape](operand: Layer.Aux[Input0, INDArrayPlaceholder.Tape])
        extends CumulativeLayer.Unary {
      type CumulativeTape = INDArraySemigroupTape with SemigroupTape with UnaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override val input = input0
        } with INDArraySemigroupTape with SemigroupTape with UnaryTape {

          val value = Transforms.log(upstream.value)

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            upstream.backward(outputDelta / upstream.value)
          }
        }
      }
    }

    final case class MultiplyDouble[Input0 <: Tape](
        operand1: Layer.Aux[Input0, INDArrayPlaceholder.Tape],
        operand2: Layer.Aux[Input0, DoublePlaceholder.Tape]
    ) extends CumulativeLayer.Binary {

      type CumulativeTape = INDArraySemigroupTape with SemigroupTape with BinaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override final val input = input0
        } with INDArraySemigroupTape with SemigroupTape with BinaryTape {

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
    final case class Dot[Input0 <: Tape](
        operand1: Layer.Aux[Input0, INDArrayPlaceholder.Tape],
        operand2: Layer.Aux[Input0, INDArrayPlaceholder.Tape]
    ) extends CumulativeLayer.Binary {

      type CumulativeTape = INDArraySemigroupTape with SemigroupTape with BinaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override final val input = input0
        } with INDArraySemigroupTape with SemigroupTape with BinaryTape {

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

    final case class Im2col[Input0 <: Tape](
        operand: Layer.Aux[Input0, INDArrayPlaceholder.Tape],
        kernel: Array[Int],
        stride: Array[Int],
        padding: Array[Int]
    ) extends CumulativeLayer.Unary {
      type CumulativeTape = INDArraySemigroupTape with SemigroupTape with UnaryTape

      type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override val input = input0
        } with INDArraySemigroupTape with SemigroupTape with UnaryTape {

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

    final case class Reshape[Input0 <: Tape](
        override val operand1: Layer.Aux[Input0, INDArrayPlaceholder.Tape],
        override val operand2: Layer.Aux[Input0, Tape.Aux[Seq[Int], (Int, Float)]])
        extends CumulativeLayer.Binary {
      override type CumulativeTape = INDArraySemigroupTape with SemigroupTape with BinaryTape

      override type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override val input = input0
        } with INDArraySemigroupTape with SemigroupTape with BinaryTape {

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

    final case class Permute[Input0 <: Tape](
        override val operand1: Layer.Aux[Input0, INDArrayPlaceholder.Tape],
        override val operand2: Layer.Aux[Input0, Tape.Aux[Seq[Int], (Int, Float)]])
        extends CumulativeLayer.Binary {
      override type CumulativeTape = INDArraySemigroupTape with SemigroupTape with BinaryTape

      override type Input = Input0

      override protected def rawForward(input0: Input): CumulativeTape = {
        new {
          override val input = input0
        } with INDArraySemigroupTape with SemigroupTape with BinaryTape {

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

//    final case class MaxPool[Input0 <: Tape](override val operand: Layer.Aux[Input0, INDArrayPlaceholder.Tape],
//                                             poolSize: (Int, Int))
//        extends CumulativeLayer.Unary {
//      override type CumulativeTape = INDArraySemigroupTape with SemigroupTape with UnaryTape
//
//      override type Input = Input0
//
//      override protected def rawForward(input0: Input): CumulativeTape = {
//        new {
//          override val input = input0
//        } with INDArraySemigroupTape with SemigroupTape with UnaryTape {
//
//          private val upstreamShape = {
//            upstream.value.shape()
//          }
//
//          private val kernelAndStrideSize: Array[Int] = toArray(poolSize)
//
//          private val preMaxPool: INDArray =
//            Convolution
//              .im2col(upstream.value, kernelAndStrideSize, kernelAndStrideSize, Array(0, 0))
//              .permute(0, 1, 4, 5, 2, 3)
//
//          private val preShape: Seq[Int] = preMaxPool.shape().toSeq
//
//          private val lastDimensionSize: Int = preShape.takeRight(2).product
//
//          private val reshapedPreMaxPool: INDArray = preMaxPool
//            .reshape(preShape.take(preShape.length - 2) :+ lastDimensionSize: _*)
//
//          override val value = reshapedPreMaxPool.max(4)
//
//          override protected def rawBackward(outputDelta: INDArray): Unit = {
//
//            val a = reshapedPreMaxPool
//            val upStreamDup = a.dup()
//            val rows = ArrayUtil.prod(a.length())
//
//            val isMax: INDArray = Nd4j.getExecutioner
//              .execAndReturn(new IsMax(upStreamDup, 4))
//              .reshape(preShape.take(preShape.length - 2) :+ poolSize._2 :+ poolSize._1: _*)
//              .permute(0, 1, 2, 4, 3, 5)
//              .reshape('c', rows, 1)
//
//            val outputDelta1d = {
//              outputDelta
//                .repeat(-1, poolSize._1)
//                .permute(1, 0, 3, 2)
//                .repeat(-1, poolSize._2)
//                .permute(1, 0, 3, 2)
//                .reshape('c', upstreamShape.product, 1)
//            }
//
//            upstream.backward(
//              isMax
//                .muliColumnVector(outputDelta1d)
//                .reshape(upstreamShape: _*)
//            )
//          }
//        }
//      }
//    }

    final case class Shape[Input0 <: Tape](operand: Layer.Aux[Input0, INDArrayPlaceholder.Tape]) extends Layer {
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

  /**
    * Returns a [[Poly.MathFunctions.max.Case Case]] that accepts a INDArray [[Layer]] and a Double [[Layer]] for the polymorphic function [[Poly.MathFunctions.max max]]
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableINDArray._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputINDArrayLayer: INDArray @Symbolic)(anotherDoubleLayer: Double @Symbolic) = {
    *   Poly.MathFunctions.max(inputINDArrayLayer,anotherDoubleLayer)
    * }
    * }}}
    */
  implicit def `max(INDArray,Double)`[Left, Right, Input <: Tape]
    : max.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Tape],
                   Layer.Aux[Input, DoublePlaceholder.Tape],
                   Layer.Aux[Input, INDArrayPlaceholder.Tape]] =
    max.at(MaxDouble(_, _))

  /**
    * Returns a [[Poly.MathMethods./.Case Case]] that accepts two INDArray [[Layer]]s.
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods./ /]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableINDArray._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputINDArrayLayer: INDArray @Symbolic)(anotherINDArrayLayer: INDArray @Symbolic) = {
    *   Poly.MathMethods./(inputINDArrayLayer,anotherINDArrayLayer)
    * }
    * }}}
    */
  implicit def `INDArray/INDArray`[Input <: Tape]
    : MathMethods./.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Tape],
                             Layer.Aux[Input, INDArrayPlaceholder.Tape],
                             Layer.Aux[Input, INDArrayPlaceholder.Tape]] = {
    MathMethods./.at { (leftLayer, rightLayer) =>
      MultiplyINDArray(leftLayer, Reciprocal(rightLayer))
    }
  }

  /**
    * Returns a [[Poly.MathMethods./.Case Case]] that accepts a Double [[Layer]] and a INDArray [[Layer]].
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods./ /]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableINDArray._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputINDArrayLayer: INDArray @Symbolic)(anotherDoubleLayer: Double @Symbolic) = {
    *   Poly.MathMethods./(inputINDArrayLayer,anotherDoubleLayer)
    * }
    * }}}
    */
  implicit def `Double/INDArray`[Input <: Tape]: MathMethods./.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Tape],
                                                                        Layer.Aux[Input, INDArrayPlaceholder.Tape],
                                                                        Layer.Aux[Input, INDArrayPlaceholder.Tape]] = {
    MathMethods./.at { (leftLayer, rightLayer) =>
      MultiplyDouble(Reciprocal(rightLayer), leftLayer)
    }
  }

  /**
    * Returns a [[Poly.MathMethods./.Case Case]] that accepts a INDArray [[Layer]]  and a Double [[Layer]].
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods./ /]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableINDArray._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputINDArrayLayer: INDArray @Symbolic)(anotherDoubleLayer: Double @Symbolic) = {
    *   Poly.MathMethods./(inputINDArrayLayer,anotherDoubleLayer)
    * }
    * }}}
    */
  implicit def `INDArray/Double`[Input <: Tape]: MathMethods./.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Tape],
                                                                        Layer.Aux[Input, DoublePlaceholder.Tape],
                                                                        Layer.Aux[Input, INDArrayPlaceholder.Tape]] = {
    MathMethods./.at { (leftLayer, rightLayer) =>
      MultiplyDouble(leftLayer, DifferentiableDouble.Layers.Reciprocal(rightLayer))
    }
  }

  /**
    * Returns a [[Poly.MathMethods.*.Case Case]] that accepts two INDArray [[Layer]]s.
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods.* *]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableINDArray._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputINDArrayLayer: INDArray @Symbolic)(anotherINDArrayLayer: INDArray @Symbolic) = {
    *   Poly.MathMethods.*(inputINDArrayLayer,anotherINDArrayLayer)
    * }
    * }}}
    */
  implicit def `INDArray*INDArray`[Input <: Tape]
    : MathMethods.*.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Tape],
                             Layer.Aux[Input, INDArrayPlaceholder.Tape],
                             Layer.Aux[Input, INDArrayPlaceholder.Tape]] = {
    MathMethods.*.at { (leftLayer, rightLayer) =>
      MultiplyINDArray(leftLayer, rightLayer)
    }
  }

  /**
    * Returns a [[Poly.MathMethods.*.Case Case]] that accepts a INDArray [[Layer]] and a Double [[Layer]].
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods.* *]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableINDArray._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputINDArrayLayer: INDArray @Symbolic)(anotherDoubleLayer: Double @Symbolic) = {
    *   Poly.MathMethods.*(inputINDArrayLayer,anotherDoubleLayer)
    * }
    * }}}
    */
  implicit def `INDArray*Double`[Input <: Tape]: MathMethods.*.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Tape],
                                                                        Layer.Aux[Input, DoublePlaceholder.Tape],
                                                                        Layer.Aux[Input, INDArrayPlaceholder.Tape]] = {
    MathMethods.*.at { (leftLayer, rightLayer) =>
      MultiplyDouble(leftLayer, rightLayer)
    }
  }

  /**
    * Returns a [[Poly.MathMethods.*.Case Case]] that accepts a Double [[Layer]] and a INDArray [[Layer]].
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods.* *]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableINDArray._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputINDArrayLayer: INDArray @Symbolic)(anotherDoubleLayer: Double @Symbolic) = {
    *   Poly.MathMethods.*(inputINDArrayLayer,anotherDoubleLayer)
    * }
    * }}}
    */
  implicit def `Double*INDArray`[Input <: Tape]: MathMethods.*.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Tape],
                                                                        Layer.Aux[Input, INDArrayPlaceholder.Tape],
                                                                        Layer.Aux[Input, INDArrayPlaceholder.Tape]] = {
    MathMethods.*.at { (leftLayer, rightLayer) =>
      MultiplyDouble(rightLayer, leftLayer)
    }
  }

  /**
    * Returns a [[Poly.MathMethods.-.Case Case]] that accepts two INDArray [[Layer]]s.
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods.- -]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableINDArray._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputINDArrayLayer: INDArray @Symbolic)(anotherINDArrayLayer: INDarray @Symbolic) = {
    *   Poly.MathMethods.-(inputINDArrayLayer,anotherINDArrayLayer)
    * }
    * }}}
    */
  implicit def `INDArray-INDArray`[Input <: Tape]
    : MathMethods.-.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Tape],
                             Layer.Aux[Input, INDArrayPlaceholder.Tape],
                             Layer.Aux[Input, INDArrayPlaceholder.Tape]] = {
    MathMethods.-.at { (leftLayer, rightLayer) =>
      PlusINDArray(leftLayer, Negative(rightLayer))
    }
  }

  /**
    * Returns a [[Poly.MathMethods.-.Case Case]] that accepts a Double [[Layer]] and a INDArray [[Layer]].
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods.- -]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableINDArray._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputINDArrayLayer: INDArray @Symbolic)(anotherDoubleLayer: Double @Symbolic) = {
    *   Poly.MathMethods.-(inputINDArrayLayer,anotherDoubleLayer)
    * }
    * }}}
    */
  implicit def `Double-INDArray`[Input <: Tape]: MathMethods.-.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Tape],
                                                                        Layer.Aux[Input, INDArrayPlaceholder.Tape],
                                                                        Layer.Aux[Input, INDArrayPlaceholder.Tape]] = {
    MathMethods.-.at { (leftLayer, rightLayer) =>
      PlusDouble(Negative(rightLayer), leftLayer)
    }
  }

  /**
    * Returns a [[Poly.MathMethods.-.Case Case]] that accepts a INDArray [[Layer]] and a Double [[Layer]].
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods.- -]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableINDArray._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputINDArrayLayer: INDArray @Symbolic)(anotherDoubleLayer: Double @Symbolic) = {
    *   Poly.MathMethods.-(inputINDArrayLayer,anotherDoubleLayer)
    * }
    * }}}
    */
  implicit def `INDArray-Double`[Input <: Tape]: MathMethods.-.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Tape],
                                                                        Layer.Aux[Input, DoublePlaceholder.Tape],
                                                                        Layer.Aux[Input, INDArrayPlaceholder.Tape]] = {
    MathMethods.-.at { (leftLayer, rightLayer) =>
      PlusDouble(leftLayer, DifferentiableDouble.Layers.Negative(rightLayer))
    }
  }

  /**
    * Returns a [[Poly.MathMethods.+.Case Case]] that accepts two INDArray [[Layer]]s.
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods.+ +]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableINDArray._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputINDArrayLayer: INDArray @Symbolic)(anotherINDArrayLayer: INDarray @Symbolic) = {
    *   Poly.MathMethods.+(inputINDArrayLayer,anotherINDArrayLayer)
    * }
    * }}}
    */
  implicit def `INDArray+INDArray`[Input <: Tape]
    : MathMethods.+.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Tape],
                             Layer.Aux[Input, INDArrayPlaceholder.Tape],
                             Layer.Aux[Input, INDArrayPlaceholder.Tape]] = {
    MathMethods.+.at { (leftLayer, rightLayer) =>
      PlusINDArray(leftLayer, rightLayer)
    }
  }

  /**
    * Returns a [[Poly.MathMethods.+.Case Case]] that accepts a INDArray [[Layer]] and a Double [[Layer]].
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods.+ +]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableINDArray._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputINDArrayLayer: INDArray @Symbolic)(anotherDoubleLayer: Double @Symbolic) = {
    *   Poly.MathMethods.+(inputINDArrayLayer,anotherDoubleLayer)
    * }
    * }}}
    */
  implicit def `INDArray+Double`[Input <: Tape]: MathMethods.+.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Tape],
                                                                        Layer.Aux[Input, DoublePlaceholder.Tape],
                                                                        Layer.Aux[Input, INDArrayPlaceholder.Tape]] = {
    MathMethods.+.at { (leftLayer, rightLayer) =>
      PlusDouble(leftLayer, rightLayer)
    }
  }

  /**
    * Returns a [[Poly.MathMethods.+.Case Case]] that accepts a Double [[Layer]] and a INDArray [[Layer]].
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathMethods.+ +]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableINDArray._
    * import com.thoughtworks.deeplearning.Symbolic
    * def myNetwork(implicit inputINDArrayLayer: INDArray @Symbolic)(anotherDoubleLayer: Double @Symbolic) = {
    *   Poly.MathMethods.+(inputINDArrayLayer,anotherDoubleLayer)
    * }
    * }}}
    */
  implicit def `Double+INDArray`[Input <: Tape]: MathMethods.+.Case.Aux[Layer.Aux[Input, DoublePlaceholder.Tape],
                                                                        Layer.Aux[Input, INDArrayPlaceholder.Tape],
                                                                        Layer.Aux[Input, INDArrayPlaceholder.Tape]] = {
    MathMethods.+.at { (leftLayer, rightLayer) =>
      PlusDouble(rightLayer, leftLayer)
    }
  }

  /**
    * Returns a [[Poly.MathFunctions.exp.Case Case]] that accepts INDArray [[Layer]]s.
    *
    * The returned `Case` is used by the polymorphic function [[Poly.MathFunctions.exp exp]],
    * which is called in [[com.thoughtworks.deeplearning.Poly.MathOps MathOps]].
    *
    * @note Importing this method will enable [[Poly.MathFunctions.exp exp]]
    *       for INDArray layers or any value able to convert to a INDArray layer
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableINDArray.`exp(INDArray)`
    * import com.thoughtworks.deeplearning.Symbolic
    * def expNetwork(implicit inputINDArrayLayer: INDArray @Symbolic) = {
    *   Poly.MathFunctions.exp(indArrayLayer)
    * }
    * }}}
    *
    * @see [[Poly.LayerPoly1]]
    */
  implicit def `exp(INDArray)`[Input <: Tape]
    : exp.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Tape], Layer.Aux[Input, INDArrayPlaceholder.Tape]] = {
    exp.at(Exp(_))
  }

  /**
    * Returns a [[Poly.MathFunctions.log.Case Case]] that accepts INDArray [[Layer]]s for the polymorphic function [[Poly.MathFunctions.log log]]
    *
    * @note Importing this method will enable [[Poly.MathFunctions.log log]]
    *       for INDArray layers or any value able to convert to a INDArray layer
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableINDArray.`log(INDArray)`
    * import com.thoughtworks.deeplearning.Symbolic
    * def logNetwork(implicit inputINDArrayLayer: INDArray @Symbolic) = {
    *   Poly.MathFunctions.log(indArrayLayer)
    * }
    * }}}
    *
    * @see [[Poly.LayerPoly1]]
    */
  implicit def `log(INDArray)`[Input <: Tape]
    : log.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Tape], Layer.Aux[Input, INDArrayPlaceholder.Tape]] = {
    log.at(Log(_))
  }

  /**
    * Returns a [[Poly.MathFunctions.abs.Case Case]] that accepts INDArray [[Layer]]s for the polymorphic function [[Poly.MathFunctions.abs abs]]
    *
    * @note Importing this method will enable [[Poly.MathFunctions.abs abs]]
    *       for INDArray layers or any value able to convert to a INDArray layer
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableINDArray.`abs(INDArray)`
    * import com.thoughtworks.deeplearning.Symbolic
    * def absNetwork(implicit inputINDArrayLayer: INDArray @Symbolic) = {
    *   Poly.MathFunctions.abs(indArrayLayer)
    * }
    * }}}
    *
    * @see [[Poly.LayerPoly1]]
    */
  implicit def `abs(INDArray)`[Input <: Tape]
    : abs.Case.Aux[Layer.Aux[Input, INDArrayPlaceholder.Tape], Layer.Aux[Input, INDArrayPlaceholder.Tape]] = {
    abs.at(Abs(_))
  }

  private def toArray(tuple2: (Int, Int)): Array[Int] = {
    val (one, two) = tuple2
    Array(one, two)
  }

  /**
    * Calculates the 2D convolution
    *
    * @param layer 4 dimensions INDArray input
    * @param weight 4 dimensions INDArray weight
    * @param bias 1 dimension bias
    * @param kernel the kernel/filter width and height
    * @param stride the stride width and height
    * @param padding the padding width and height
    * @return convolution result
    */
  def conv2d[Layer, Weight, Bias, Input <: Tape](layer: Layer,
                                                 weight: Weight,
                                                 bias: Bias,
                                                 kernel: (Int, Int),
                                                 stride: (Int, Int),
                                                 padding: (Int, Int))(
      implicit layerToLayer: ToLayer.Aux[Layer, Input, INDArray, INDArray],
      weightToLayer: ToLayer.Aux[Weight, Input, INDArray, INDArray],
      biasToLayer: ToLayer.Aux[Bias, Input, INDArray, INDArray]): Layer.Aux[Input, INDArrayPlaceholder.Tape] = {
    val layerOfInput = layerToLayer(layer)
    val shapeOfOperand = Shape(layerOfInput)
    val count = DifferentiableSeq.Layers.Get(shapeOfOperand, 0)
    val depth = DifferentiableSeq.Layers.Get(shapeOfOperand, 1)
    val y_axis = DifferentiableSeq.Layers.Get(shapeOfOperand, 2)
    val x_axis = DifferentiableSeq.Layers.Get(shapeOfOperand, 3)
    val kernelNumber = DifferentiableSeq.Layers.Get(Shape(weightToLayer(weight)), 0)

    //input
    //  .im2col(Array(KernelSize, KernelSize),
    //    Array(Stride, Stride),
    //    Array(Padding, Padding))
    val col: Layer.Aux[Input, Tape.Aux[INDArray, INDArray]] =
      Im2col(layerOfInput, toArray(kernel), toArray(stride), toArray(padding))

    //permute(0, 4, 5, 1, 2, 3)
    val permutedCol: Layer.Aux[Input, Tape.Aux[INDArray, INDArray]] = Permute(col, Literal(Seq(0, 4, 5, 1, 2, 3)))

    val depthKernelKernel: Layer.Aux[Input, Tape.Aux[Int, Float]] =
      Times(
        Times(depth, Literal(kernel._1)),
        Literal(kernel._2)
      )

    val countXaxisYaxis: Layer.Aux[Input, Tape.Aux[Int, Float]] = Times(Times(count, y_axis), x_axis)

    val aSeq: Seq[Layer.Aux[Input, Tape.Aux[Int, Float]]] = Seq(countXaxisYaxis, depthKernelKernel)

    val reshapeOperandTo: Layer.Aux[Input, Tape.Aux[Seq[Int], (Int, Float)]] = DifferentiableSeq.Layers.ToSeq(aSeq)

    //reshape(imageCount * inputSizeY * inputSizeX,(depth * KernelSize * KernelSize).toLayer)
    val operandCol2d = Reshape(permutedCol, reshapeOperandTo)

    val bSeq: Seq[Layer.Aux[Input, Tape.Aux[Int, Float]]] = Seq(kernelNumber, depthKernelKernel)

    val reshapeWeightTo: Layer.Aux[Input, Tape.Aux[Seq[Int], (Int, Float)]] = DifferentiableSeq.Layers.ToSeq(bSeq)

    //weight.reshape(kernelNumber, KernelSize * KernelSize * depth)
    val reshapedWeight = Reshape(weight, reshapeWeightTo)

    //permute(1, 0)
    val permutedWeight = Permute(reshapedWeight, Literal(Seq(1, 0)))

    val dotResult = Dot(operandCol2d, permutedWeight)

    val plusResult = PlusINDArray(dotResult, biasToLayer(bias))

    val SeqOfCountYaxisXaxisKernelNumber: Seq[Layer.Aux[Input, Tape.Aux[Int, Float]]] =
      Seq(count, y_axis, x_axis, kernelNumber)

    val reshapeResultTo: Layer.Aux[Input, Tape.Aux[Seq[Int], (Int, Float)]] =
      DifferentiableSeq.Layers.ToSeq(SeqOfCountYaxisXaxisKernelNumber)

    //reshape(imageCount, inputSizeY, inputSizeX, kernelNumber.toLayer)
    val reshapedResult = Reshape(plusResult, reshapeResultTo)

    val permuteResultTo = Literal(Seq(0, 3, 1, 2))

    //permute(0, 3, 1, 2)
    Permute(reshapedResult, permuteResultTo)
  }

  final class INDArrayLayerOps[Input <: Tape](operand: Layer.Aux[Input, INDArrayPlaceholder.Tape]) {

    // TODO: Considering if rename this method to `matmul`
    def dot[Right](right: Right)(implicit rightToLayer: ToLayer.Aux[Right, Input, INDArray, INDArray])
      : Layer.Aux[Input, INDArrayPlaceholder.Tape] = {
      Dot(operand, rightToLayer(right))
    }

    /**
      * Im2col ops
      * @param kernel kernel size / filter size
      * @param stride stride size
      * @param padding padding size
      */
    def im2col(kernel: Array[Int],
               stride: Array[Int],
               padding: Array[Int]): Layer.Aux[Input, INDArrayPlaceholder.Tape] = {
      Im2col(operand, kernel, stride, padding)
    }

    /**
      * @usecase def reshape(newShape: Layer.Aux[Input, Tape.Aux[Int, Float]]*): Layer.Aux[Input, INDArrayPlaceholder.Tape] = ???
      * @usecase def reshape(newShape: Int*): Layer.Aux[Input, INDArrayPlaceholder.Tape] = ???
      */
    def reshape[Element](newShape: Element*)(
        implicit toLayer: ToLayer.Aux[Element, Input, Int, Float]): Layer.Aux[Input, INDArrayPlaceholder.Tape] = {
      Reshape(operand, DifferentiableSeq.Layers.ToSeq(newShape.map(toLayer.apply(_))))
    }

    /**
      * @usecase def permute(newShape: Layer.Aux[Input, Tape.Aux[Int, Float]]*): Layer.Aux[Input, INDArrayPlaceholder.Tape] = ???
      * @usecase def permute(newShape: Int*): Layer.Aux[Input, INDArrayPlaceholder.Tape] = ???
      */
    def permute[Element](newShape: Element*)(
        implicit toLayer: ToLayer.Aux[Element, Input, Int, Float]): Layer.Aux[Input, INDArrayPlaceholder.Tape] = {
      Permute(operand, DifferentiableSeq.Layers.ToSeq(newShape.map(toLayer.apply(_))))
    }

//    def maxPool(poolSize: (Int, Int)): Layer.Aux[Input, INDArrayPlaceholder.Tape] = {
//      MaxPool(operand, poolSize)
//    }

    /**
      * Returns shape of INDArray
      */
    def shape: Layer.Aux[Input, Tape.Aux[Seq[Int], (Int, Float)]] = {
      Shape(operand)
    }

    /**
      * Returns opposite of all elements
      */
    def unary_- : Layer.Aux[Input, INDArrayPlaceholder.Tape] = {
      Negative(operand)
    }

    def toSeq: Layer.Aux[Input, Tape.Aux[Seq[Seq[Double]], (Int, (Int, Double))]] = {
      ToSeq(operand)
    }

    /**
      * Returns sum of all elements of INDArray
      */
    def sum: Layer.Aux[Input, DoublePlaceholder.Tape] = {
      ReduceSum(operand)
    }

    /**
      * Returns mean of all elements of INDArray
      */
    def mean: Layer.Aux[Input, DoublePlaceholder.Tape] = {
      ReduceMean(operand)
    }

    /**
      * Returns sum dimensions of INDArray,will Returns an INDArrayPlaceholder
      */
    def sum(dimensions: Int*): Layer.Aux[Input, INDArrayPlaceholder.Tape] = {
      Sum(operand, dimensions)
    }

  }

  /**
    * Implicitly converts any layer to [[INDArrayLayerOps]], which enables common methods for INDArray layers.

    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableINDArray._
    * }}}
    */
  implicit def toINDArrayLayerOps[From, Input <: Tape, OutputData, OutputDelta](from: From)(
      implicit toLayer: ToLayer.Aux[From, Input, OutputData, OutputDelta],
      constrait: Layer.Aux[Input, Tape.Aux[OutputData, OutputDelta]] <:< Layer.Aux[Input, Tape.Aux[INDArray, INDArray]]
  ): INDArrayLayerOps[Input] = {
    new INDArrayLayerOps(constrait(toLayer(from)))
  }

  // TODO: Support Array for better performance.
  final class ToINDArrayLayerOps[Input <: Tape](layerVector: Seq[Seq[Layer.Aux[Input, Tape.Aux[Double, Double]]]]) {
    def toINDArray: Layer.Aux[Input, INDArrayPlaceholder.Tape] = ToINDArray(layerVector)
  }

  implicit def toToINDArrayLayerOps[Element, Input <: Tape](layerVector: Seq[Seq[Element]])(
      implicit toLayer: ToLayer.OfPlaceholder[Element, Input, DoublePlaceholder]): ToINDArrayLayerOps[Input] = {
    new ToINDArrayLayerOps(layerVector.map(_.map(toLayer(_))))
  }

  implicit final class INDArrayOps(ndArray: INDArray) {
    def toWeight[InputData, InputDelta](
        implicit inputType: Placeholder[InputData, InputDelta],
        optimizerFactory: OptimizerFactory): Layer.Aux[Tape.Aux[InputData, InputDelta], INDArrayPlaceholder.Tape] = {
      Weight(ndArray)
    }
  }

  implicit def indArrayToLiteral: ToLiteral.Aux[INDArray, INDArray, INDArray] = ToLiteral.fromData

  /**
    * @see [[com.thoughtworks.deeplearning.DifferentiableAny.Trainable Trainable]]
    */
  implicit def indArrayTrainable: Trainable[INDArray, INDArray] = new Trainable[INDArray, INDArray] {
    override def apply(data: INDArray): INDArray = Nd4j.ones(data.shape(): _*)
  }

}
