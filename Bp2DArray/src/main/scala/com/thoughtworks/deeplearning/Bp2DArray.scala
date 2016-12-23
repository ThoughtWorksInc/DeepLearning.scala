package com.thoughtworks.deeplearning

import cats.implicits._
import cats.{Applicative, Eval, Semigroup, Traverse}
import com.thoughtworks.deeplearning.Layer.{Aux, Batch, CloseableOnce}
import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.Bp2DArray.Layers._
import com.thoughtworks.deeplearning.Bp2DArray.Optimizers._
import com.thoughtworks.deeplearning.BpDouble._
import com.thoughtworks.deeplearning.Lift.Layers.Literal
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

    override type Data = INDArray

    override type Delta = INDArray

    protected final def semigroup = new Semigroup[Delta] {
      override def combine(x: Delta, y: Delta): Delta = x + y
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
  type Bp2DArray = Placeholder[INDArray, INDArray]

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
            val aValue = upstream1.value
            val bValue = upstream2.value
            val Array(aRows, aColumns) = aValue.shape()
            val Array(bRows, bColumns) = bValue.shape()
            val newShape = Array(math.max(aRows, bRows), math.max(aColumns, bColumns))
            aValue.broadcast(newShape: _*) * bValue.broadcast(newShape: _*)
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

    final case class MaxBpDouble[Input0 <: Batch](
        operand1: Layer.Aux[Input0, Bp2DArray#Batch],
        operand2: Layer.Aux[Input0, DoubleBackProgationType.Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch {

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
            val aValue = upstream1.value
            val bValue = upstream2.value
            val Array(aRows, aColumns) = aValue.shape()
            val Array(bRows, bColumns) = bValue.shape()
            val newShape =
              Array(math.max(aRows, bRows), math.max(aColumns, bColumns))
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
    final case class ToSeq[Input0 <: Batch](operand: Layer.Aux[Input0, Bp2DArray#Batch]) extends BufferedLayer.Unary {
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

          override def backward(delta: Delta): Unit = {
            synchronized {
              val (i, (j, value)) = delta
              // Cannot use += because of https://issues.scala-lang.org/browse/SI-10021
              upstreamDelta(i, j) = upstreamDelta(i, j) + value
            }
          }

          override val value: Data = {
            val ndarray = upstream.value
            val doubleArray = ndarray.data.asDouble()
            for (i <- (0 until ndarray.rows).view) yield {
              doubleArray.view(i * ndarray.columns, (i + 1) * ndarray.columns)
            }
          }
        }
      }
    }

    final case class Weight(var value: INDArray)(implicit optimizer: Optimizer)
        extends Layer
        with TwoDArraySemigroupBatch {
      override type Input = Batch
      override type Output = Batch.Aux[Data, Delta]

      override def addReference() = this

      override def forward(any: Input) = this

      override def backward(delta: Delta): Unit = {
        synchronized {
          value = optimizer.updateNDArray(value, delta)
        }
      }

      override def close(): Unit = {}

    }

    final case class ToBp2DArray[Input0 <: Batch](operands: Seq[Seq[Layer.Aux[Input0, Batch.Aux[Double, Double]]]])
        extends Layer {

      type Input = Input0

      final class Output private[ToBp2DArray] (upstreams: Seq[Seq[Batch.Aux[Double, Double]]])
          extends TwoDArraySemigroupBatch
          with CloseableOnce {
        override def backward(delta: INDArray): Unit = {
          for ((row, i) <- upstreams.view.zipWithIndex; (upstream, j) <- row.zipWithIndex) {
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

    final case class Sum[Input0 <: Batch](operand: Layer.Aux[Input0, Bp2DArray#Batch], dimensions: Seq[Int])
        extends BufferedLayer.Unary {
      type BufferedBatch = TwoDArraySemigroupBatch with SemigroupBatch with UnaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override val input = input0
        } with TwoDArraySemigroupBatch with SemigroupBatch with UnaryBatch {

          val value = upstream.value.sum(dimensions: _*)

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            val a = upstream.value
            upstream.backward(outputDelta.broadcast(a.shape: _*))
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

          val value = (upstream.value: INDArray).sumT

          override protected def rawBackward(outputDelta: Double): Unit = {
            upstream.backward(Nd4j.valueArrayOf(upstream.value.shape(), outputDelta))
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

          val value = upstream.value rdiv 1.0

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            val upstreamValue: INDArray = upstream.value
            upstream.backward(-outputDelta / (upstreamValue * upstreamValue))
          }
        }
      }
    }

    final case class PlusBpDouble[Input0 <: Batch](
        operand1: Layer.Aux[Input0, Bp2DArray#Batch],
        operand2: Layer.Aux[Input0, DoubleBackProgationType.Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch {
          val value = (upstream1.value: INDArray) + upstream2.value

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            upstream1.backward(outputDelta)
            upstream2.backward(outputDelta.sumT)
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

          val value = -upstream.value

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            upstream.backward(-outputDelta)
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
          val value = Transforms.exp(upstream.value)

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            upstream.backward(value * outputDelta)
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

          val value = Transforms.log(upstream.value)

          override protected def rawBackward(outputDelta: INDArray): Unit = {
            upstream.backward(outputDelta / upstream.value)
          }
        }
      }
    }

    final case class MultiplyBpDouble[Input0 <: Batch](
        operand1: Layer.Aux[Input0, Bp2DArray#Batch],
        operand2: Layer.Aux[Input0, DoubleBackProgationType.Batch]
    ) extends BufferedLayer.Binary {

      type BufferedBatch = TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch

      type Input = Input0

      override protected def rawForward(input0: Input): BufferedBatch = {
        new {
          override final val input = input0
        } with TwoDArraySemigroupBatch with SemigroupBatch with BinaryBatch {

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

  }

  import Layers._

  implicit def `max(Bp2DArray,Double)`[Left, Right, Input <: Batch]
    : max.Case.Aux[Layer.Aux[Input, Bp2DArray#Batch],
                   Layer.Aux[Input, DoubleBackProgationType.Batch],
                   Layer.Aux[Input, Bp2DArray#Batch]] =
    max.at(MaxBpDouble(_, _))

  implicit def `Bp2DArray/Bp2DArray`[Input <: Batch]: MathMethods./.Case.Aux[Layer.Aux[Input, Bp2DArray#Batch],
                                                                             Layer.Aux[Input, Bp2DArray#Batch],
                                                                             Layer.Aux[Input, Bp2DArray#Batch]] = {
    MathMethods./.at { (leftLayer, rightLayer) =>
      MultiplyBp2DArray(leftLayer, Reciprocal(rightLayer))
    }
  }

  implicit def `BpDouble/Bp2DArray`[Input <: Batch]
    : MathMethods./.Case.Aux[Layer.Aux[Input, DoubleBackProgationType.Batch],
                             Layer.Aux[Input, Bp2DArray#Batch],
                             Layer.Aux[Input, Bp2DArray#Batch]] = {
    MathMethods./.at { (leftLayer, rightLayer) =>
      MultiplyBpDouble(Reciprocal(rightLayer), leftLayer)
    }
  }

  implicit def `Bp2DArray/BpDouble`[Input <: Batch]
    : MathMethods./.Case.Aux[Layer.Aux[Input, Bp2DArray#Batch],
                             Layer.Aux[Input, DoubleBackProgationType.Batch],
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

  implicit def `Bp2DArray*BpDouble`[Input <: Batch]
    : MathMethods.*.Case.Aux[Layer.Aux[Input, Bp2DArray#Batch],
                             Layer.Aux[Input, DoubleBackProgationType.Batch],
                             Layer.Aux[Input, Bp2DArray#Batch]] = {
    MathMethods.*.at { (leftLayer, rightLayer) =>
      MultiplyBpDouble(leftLayer, rightLayer)
    }
  }

  implicit def `BpDouble*Bp2DArray`[Input <: Batch]
    : MathMethods.*.Case.Aux[Layer.Aux[Input, DoubleBackProgationType.Batch],
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

  implicit def `BpDouble-Bp2DArray`[Input <: Batch]
    : MathMethods.-.Case.Aux[Layer.Aux[Input, DoubleBackProgationType.Batch],
                             Layer.Aux[Input, Bp2DArray#Batch],
                             Layer.Aux[Input, Bp2DArray#Batch]] = {
    MathMethods.-.at { (leftLayer, rightLayer) =>
      PlusBpDouble(Negative(rightLayer), leftLayer)
    }
  }

  implicit def `Bp2DArray-BpDouble`[Input <: Batch]
    : MathMethods.-.Case.Aux[Layer.Aux[Input, Bp2DArray#Batch],
                             Layer.Aux[Input, DoubleBackProgationType.Batch],
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

  implicit def `Bp2DArray+BpDouble`[Input <: Batch]
    : MathMethods.+.Case.Aux[Layer.Aux[Input, Bp2DArray#Batch],
                             Layer.Aux[Input, DoubleBackProgationType.Batch],
                             Layer.Aux[Input, Bp2DArray#Batch]] = {
    MathMethods.+.at { (leftLayer, rightLayer) =>
      PlusBpDouble(leftLayer, rightLayer)
    }
  }

  implicit def `BpDouble+Bp2DArray`[Input <: Batch]
    : MathMethods.+.Case.Aux[Layer.Aux[Input, DoubleBackProgationType.Batch],
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

    def toSeq: Layer.Aux[Input, Batch.Aux[Seq[Seq[Double]], (Int, (Int, Double))]] = {
      ToSeq(differentiable)
    }

  }

  implicit def to2DArrayLayerOps[From, Input <: Batch](from: From)(
      implicit toLayer: ToLayer.OfType[From, Input, Bp2DArray]
  ): TwoDArrayLayerOps[Input] = {
    new TwoDArrayLayerOps(toLayer(from))
  }

  // TODO: Support Array for better performance.
  final class To2DArrayLayerOps[Input <: Batch](layerVector: Seq[Seq[Layer.Aux[Input, Batch.Aux[Double, Double]]]]) {
    def toBp2DArray: Layer.Aux[Input, Bp2DArray#Batch] = ToBp2DArray(layerVector)
  }

  implicit def toTo2DArrayLayerOps[Element, Input <: Batch](layerVector: Seq[Seq[Element]])(
      implicit toLayer: ToLayer.OfType[Element, Input, DoubleBackProgationType]): To2DArrayLayerOps[Input] = {
    new To2DArrayLayerOps(layerVector.view.map(_.view.map(toLayer(_))))
  }

  implicit final class INDArrayOps(ndArray: INDArray) {
    def toWeight[InputData, InputDelta](
                                         implicit inputType: Placeholder[InputData, InputDelta],
                                         optimizer: Optimizer): Layer.Aux[Batch.Aux[InputData, InputDelta], Bp2DArray#Batch] = {
      Weight(ndArray)
    }
  }

  implicit def liftINDArray: Lift.Aux[INDArray, INDArray, INDArray] = Lift.fromData[INDArray, INDArray]

}
