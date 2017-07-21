package com.thoughtworks.deeplearning.plugins

import java.io.{PrintStream, PrintWriter}

import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply}
import com.thoughtworks.raii.asynchronous.{Do, ParallelDo}
import com.thoughtworks.raii.asynchronous.Do._
import org.nd4j.linalg.api.ndarray.INDArray

import scala.annotation.meta.getter
import scalaz.syntax.all._
import scalaz.Tags.Parallel
import scalaz.{@@, Apply, Isomorphism, IsomorphismSemigroup, Semigroup}
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms
import DeepLearning.ops._
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.ArrayUtil

import scala.concurrent.ExecutionContext
import com.thoughtworks.future.continuation.{Continuation, ParallelContinuation, UnitContinuation}

import scala.util.control.NoStackTrace

object INDArrayLayers {

  final case class MultipleException(throwableSet: Set[Throwable]) extends Exception("Multiple exceptions found") {
    override def toString: String = throwableSet.mkString("\n")

    override def printStackTrace(): Unit = {
      for (throwable <- throwableSet) {
        throwable.printStackTrace()
      }
    }

    override def printStackTrace(s: PrintStream): Unit = {
      for (throwable <- throwableSet) {
        throwable.printStackTrace(s)
      }
    }

    override def printStackTrace(s: PrintWriter): Unit = {
      for (throwable <- throwableSet) {
        throwable.printStackTrace(s)
      }
    }

    override def getStackTrace: Array[StackTraceElement] = synchronized {
      super.getStackTrace match {
        case null =>
          setStackTrace(throwableSet.flatMap(_.getStackTrace)(collection.breakOut))
          super.getStackTrace
        case stackTrace =>
          stackTrace
      }
    }

    override def fillInStackTrace(): this.type = {
      this
    }

  }

  // Workaround for https://github.com/deeplearning4j/nd4j/issues/1869
  private[plugins] implicit final class Nd4jIssues1869Workaround(indArray: INDArray) {
    def broadcastFix(outputShape: Int*): INDArray = {
      val currentShape = indArray.shape.padTo(outputShape.length, 1)
      currentShape.indices.foldLeft(indArray.reshape(currentShape: _*)) { (indArray, i) =>
        val o = outputShape(i)
        if (o != 1 && o != currentShape(i)) {
          currentShape(i) = o
          indArray.broadcast(currentShape: _*)
        } else {
          indArray
        }
      }
    }
  }
}

/** A plugin that provides differentiable operators
  * on neural networks whose [[DeepLearning.Data Data]] and [[DeepLearning.Delta Delta]] is [[org.nd4j.linalg.api.ndarray.INDArray]].
  *
  * @note By default, the computation in a [[INDArrayLayer]] will re-evaluate  again and again
  *       if the `INDArrayLayer` is used by multiple other operations.
  *
  *       This behavior is very inefficient if there is are diamond dependencies in a neural network.
  *       It's wise to use [[CumulativeINDArrayLayers]] instead of this `INDArrayLayers` in such neural network.
  *
  * @author 杨博 (Yang Bo)
  */
trait INDArrayLayers extends DoubleLayers with DoubleLiterals with ImplicitsSingleton {
  import INDArrayLayers._

  // TODO: Add a test for this method and auto-broadcasting on n-dimension arrays for n > 2
  private def sumAs(outputDeltaValue: INDArray, shape: Array[Int]) = {
    val singleElementDimension = (shape: Seq[Int]).view.zip(outputDeltaValue.shape).zipWithIndex.collect {
      case ((1, originSize), dimension) if originSize > 1 => dimension
    }
    if (singleElementDimension.isEmpty) {
      outputDeltaValue
    } else {
      outputDeltaValue.sum(singleElementDimension.force: _*).reshape(shape: _*)
    }
  }

  private def autoBroadcastShape(shape1: Array[Int], shape2: Array[Int]): Array[Int] = {
    require(shape1.length == shape2.length)
    shape1.zip(shape2).map {
      case (1, bSize)                       => bSize
      case (aSize, 1)                       => aSize
      case (aSize, bSize) if aSize == bSize => aSize
    }
  }

  @transient
  implicit private lazy val unitFutureSemigroup: Semigroup[UnitContinuation[Unit]] = {
    Parallel.unsubst(
      Semigroup.liftSemigroup[ParallelContinuation, Unit](
        Continuation.continuationParallelApplicative,
        scalaz.std.anyVal.unitInstance
      )
    )
  }

  @transient
  private lazy val doParallelApplicative = ParallelDo.doParallelApplicative(new Semigroup[Throwable] {
    override def append(f1: Throwable, f2: => Throwable): Throwable =
      f1 match {
        case MultipleException(exceptionSet1) =>
          f2 match {
            case MultipleException(exceptionSet2) => MultipleException(exceptionSet1 ++ exceptionSet2)
            case _: Throwable                     => MultipleException(exceptionSet1 + f2)
          }
        case _: Throwable =>
          f2 match {
            case MultipleException(exceptionSet2) => MultipleException(exceptionSet2 + f1)
            case _: Throwable                     => MultipleException(Set(f1, f2))
          }
      }
  })

  private def parallelApply2[A, B, C](doA: Do[A], doB: Do[B])(f: (A, B) => C): Do[C] = {
    Parallel.unwrap(doParallelApplicative.apply2(Parallel(doA), Parallel(doB))(f))
  }

  /** The [[scala.concurrent.ExecutionContext]] used internally in plugins. */
  @inject
  implicit protected def deepLearningExecutionContext: ExecutionContext

  def dot[Operand0, Operand1, Out <: INDArrayLayer](operand0: Operand0, operand1: Operand1)(
      implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
      deepLearning1: DeepLearning.Aux[Operand1, INDArray, INDArray],
      layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]): Out = {
    INDArrayLayer.binary(operand0, operand1) { (data0: INDArray, data1: INDArray) =>
      val outputData = data0 dot data1
      val delta0 = { outputDelta: INDArray =>
        outputDelta dot data1.T
      }
      val delta1 = { outputDelta: INDArray =>
        data0.T dot outputDelta
      }
      (outputData, delta0, delta1)
    }
  }
  trait ImplicitsApi extends super[DoubleLiterals].ImplicitsApi with super[DoubleLayers].ImplicitsApi {

    /** An implicit wrapper that adds extension methods for differentiable n-dimensional array types
      * that support the [[DeepLearning]] type class.
      */
    implicit final class INDArrayLayerOps[Operand0](operand0: Operand0)(
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray]) {

      /** @usecase def unary_- : INDArrayLayer = ???
        */
      def unary_-[Out <: INDArrayLayer](
          implicit layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]): Out = {
        INDArrayLayer.unary(operand0) { data0: INDArray =>
          val outputData = -data0
          val delta0 = { outputDelta: INDArray =>
            -outputDelta
          }
          (outputData, delta0)
        }
      }

      /** @usecase def mean: DoubleLayer = ???
        */
      def mean[Out <: DoubleLayer](
          implicit layerImplicits: ImplicitApply.Aux[doublePartialApplyRawForward.Rest, Out]
      ): Out = {
        DoubleLayer(operand0.forward.flatMap { tape =>
          Operators./(operand0.sum, tape.data.length.toDouble).forward
        })
      }

      /** @usecase def mean(dimensions: Int*): INDArrayLayer = ???
        */
      def mean[Out <: INDArrayLayer](dimensions: Int*)(
          implicit layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]
      ): Out = {
        INDArrayLayer(operand0.forward.flatMap { tape =>
          val shape = tape.data.shape
          Operators
            ./(operand0.sum(dimensions: _*), dimensions.map(shape(_).toDouble).product)(`INDArray/Double`)
            .forward
        })
      }

      @deprecated(message = "Use `mean` instead.", since = "2.0.0")
      def meanT[Out <: DoubleLayer](
          implicit layerImplicits: ImplicitApply.Aux[doublePartialApplyRawForward.Rest, Out]
      ): Out = {
        mean
      }

      /** @usecase def sum(dimensions: Int*): INDArrayLayer = ???
        */
      def sum[Out <: INDArrayLayer](dimensions: Int*)(
          implicit layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]): Out = {
        INDArrayLayer.unary(operand0) { data0: INDArray =>
          val shape0 = data0.shape
          val outputData = data0.sum(dimensions: _*)
          val delta0 = { outputDelta: INDArray =>
            outputDelta.broadcast(shape0: _*)
          }
          (outputData, delta0)
        }
      }

      /** @usecase def sum: DoubleLayer = ???
        */
      def sum[Out <: DoubleLayer](
          implicit layerImplicits: ImplicitApply.Aux[doublePartialApplyRawForward.Rest, Out]): Out = {
        DoubleLayer.unary(operand0) { data0: INDArray =>
          val shape0 = data0.shape
          val outputData = data0.sumT
          val delta0 = { outputDelta: Double =>
            Nd4j.valueArrayOf(shape0, outputDelta)
          }
          (outputData, delta0)
        }
      }

      @deprecated(message = "Use `sum` instead.", since = "2.0.0")
      def sumT[Out <: DoubleLayer](
          implicit layerImplicits: ImplicitApply.Aux[doublePartialApplyRawForward.Rest, Out]): Out =
        sum

      /** @usecase def permute(dimensions: Int*): INDArrayLayer = ???
        */
      def permute[Out <: INDArrayLayer](dimensions: Int*)(
          implicit layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]): Out = {
        INDArrayLayer.unary(operand0) { data0: INDArray =>
          val shape0 = data0.shape
          val outputData = data0.permute(dimensions: _*)
          val delta0 = { outputDelta: INDArray =>
            outputDelta.permute(shape0.indices.map(dimensions.indexOf): _*)
          }
          (outputData, delta0)
        }
      }

      /** @usecase def reshape(dimensions: Int*): INDArrayLayer = ???
        */
      def reshape[Out <: INDArrayLayer](dimensions: Int*)(
          implicit layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]): Out = {
        INDArrayLayer.unary(operand0) { data0: INDArray =>
          val shape0 = data0.shape
          val outputData = data0.reshape(dimensions: _*)
          val delta0 = { outputDelta: INDArray =>
            outputDelta.reshape(shape0: _*)
          }
          (outputData, delta0)
        }
      }

      /** @usecase def dot(operand1: INDArray): INDArrayLayer = ???
        * @usecase def dot(operand1: INDArrayLayer): INDArrayLayer = ???
        * @usecase def dot(operand1: INDArrayWeights#INDArrayWeight): INDArrayLayer = ???
        */
      def dot[Operand1, Out <: INDArrayLayer](operand1: Operand1)(
          implicit deepLearning1: DeepLearning.Aux[Operand1, INDArray, INDArray],
          layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]): Out = {
        INDArrayLayers.this
          .dot[Operand0, Operand1, Out](operand0, operand1)(deepLearning0, deepLearning1, layerImplicits)
      }
    }

    implicit def `INDArray+INDArray`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
        deepLearning1: DeepLearning.Aux[Operand1, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.+.at[Operand0, Operand1] {
        INDArrayLayer.binary(_, _) { (data0: INDArray, data1: INDArray) =>
          val shape0 = data0.shape
          val shape1 = data1.shape
          val outputShape = autoBroadcastShape(shape0, shape1)
          val broadcastData0 = data0.broadcastFix(outputShape: _*)
          val broadcastData1 = data1.broadcastFix(outputShape: _*)
          val outputData = broadcastData0 + broadcastData1
          val delta0 = { (outputDelta: INDArray) =>
            sumAs(outputDelta, shape0)
          }
          val delta1 = { (outputDelta: INDArray) =>
            sumAs(outputDelta, shape1)
          }
          (outputData, delta0, delta1)
        }
      }
    }

    implicit def `INDArray+Double`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
        deepLearning1: DeepLearning.Aux[Operand1, Double, Double],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.+.at[Operand0, Operand1] {
        INDArrayLayer.binary(_, _) { (data0: INDArray, data1: Double) =>
          val outputData = data0 + data1
          val delta0 = { (outputDelta: INDArray) =>
            outputDelta
          }
          val delta1 = { (outputDelta: INDArray) =>
            outputDelta.sumT
          }
          (outputData, delta0, delta1)
        }
      }
    }

    implicit def `Double+INDArray`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Double, Double],
        deepLearning1: DeepLearning.Aux[Operand1, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.+.at[Operand0, Operand1] { (operand0, operand1) =>
        `INDArray+Double`[Operand1, Operand0, Out].apply(operand1, operand0)
      }
    }

    implicit def `INDArray-INDArray`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
        deepLearning1: DeepLearning.Aux[Operand1, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.-.at[Operand0, Operand1] {
        INDArrayLayer.binary(_, _) { (data0: INDArray, data1: INDArray) =>
          val shape0 = data0.shape
          val shape1 = data1.shape
          val outputShape = autoBroadcastShape(shape0, shape1)
          val broadcastData0 = data0.broadcastFix(outputShape: _*)
          val broadcastData1 = data1.broadcastFix(outputShape: _*)
          val outputData = broadcastData0 - broadcastData1
          val delta0 = { (outputDelta: INDArray) =>
            sumAs(outputDelta, shape0)
          }
          val delta1 = { (outputDelta: INDArray) =>
            -sumAs(outputDelta, shape1)
          }
          (outputData, delta0, delta1)
        }
      }
    }

    implicit def `INDArray-Double`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
        deepLearning1: DeepLearning.Aux[Operand1, Double, Double],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.-.at[Operand0, Operand1] {
        INDArrayLayer.binary(_, _) { (data0: INDArray, data1: Double) =>
          val outputData = data0 - data1
          val delta0 = { (outputDelta: INDArray) =>
            outputDelta
          }
          val delta1 = { (outputDelta: INDArray) =>
            -outputDelta.sumT
          }
          (outputData, delta0, delta1)
        }
      }
    }

    implicit def `Double-INDArray`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Double, Double],
        deepLearning1: DeepLearning.Aux[Operand1, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.-.at[Operand0, Operand1] {
        INDArrayLayer.binary(_, _) { (data0: Double, data1: INDArray) =>
          val outputData = data1 rsub data0

          val delta0 = { (outputDelta: INDArray) =>
            outputDelta.sumT
          }
          val delta1 = { (outputDelta: INDArray) =>
            -outputDelta
          }
          (outputData, delta0, delta1)
        }
      }
    }

    implicit def `INDArray*INDArray`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
        deepLearning1: DeepLearning.Aux[Operand1, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.*.at[Operand0, Operand1] {
        INDArrayLayer.binary(_, _) { (data0: INDArray, data1: INDArray) =>
          val shape0 = data0.shape
          val shape1 = data1.shape
          val outputShape = autoBroadcastShape(shape0, shape1)
          val broadcastData0 = data0.broadcastFix(outputShape: _*)
          val broadcastData1 = data1.broadcastFix(outputShape: _*)
          val outputData = broadcastData0 * broadcastData1
          val delta0 = { outputDelta: INDArray =>
            sumAs(outputDelta * broadcastData1, shape0)
          }
          val delta1 = { outputDelta: INDArray =>
            sumAs(outputDelta * broadcastData0, shape1)
          }
          (outputData, delta0, delta1)
        }
      }
    }

    implicit def `INDArray*Double`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
        deepLearning1: DeepLearning.Aux[Operand1, Double, Double],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.*.at[Operand0, Operand1] {
        INDArrayLayer.binary(_, _) { (data0: INDArray, data1: Double) =>
          val outputData = data0 * data1
          val delta0 = { (outputDelta: INDArray) =>
            outputDelta * data1
          }
          val delta1 = { (outputDelta: INDArray) =>
            (data0 * outputDelta).sumT
          }
          (outputData, delta0, delta1)
        }
      }
    }

    implicit def `Double*INDArray`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Double, Double],
        deepLearning1: DeepLearning.Aux[Operand1, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.*.at[Operand0, Operand1] { (operand0, operand1) =>
        `INDArray*Double`[Operand1, Operand0, Out].apply(operand1, operand0)
      }
    }

    implicit def `INDArray/INDArray`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
        deepLearning1: DeepLearning.Aux[Operand1, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators./.at[Operand0, Operand1] {
        INDArrayLayer.binary(_, _) { (data0: INDArray, data1: INDArray) =>
          val shape0 = data0.shape
          val shape1 = data1.shape
          val outputShape = autoBroadcastShape(shape0, shape1)
          val broadcastData0 = data0.broadcastFix(outputShape: _*)
          val broadcastData1 = data1.broadcastFix(outputShape: _*)
          val outputData = broadcastData0 / broadcastData1
          val delta0 = { outputDelta: INDArray =>
            sumAs(outputDelta / broadcastData1, shape0)
          }
          val delta1 = { outputDelta: INDArray =>
            sumAs(-outputDelta * broadcastData0 / (data1 * data1).broadcastFix(outputDelta.shape: _*), shape1)
          }
          (outputData, delta0, delta1)
        }
      }
    }

    implicit def `INDArray/Double`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
        deepLearning1: DeepLearning.Aux[Operand1, Double, Double],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators./.at[Operand0, Operand1] {
        INDArrayLayer.binary(_, _) { (data0: INDArray, data1: Double) =>
          val outputData = data0 / data1
          val delta0 = { (outputDelta: INDArray) =>
            outputDelta / data1
          }
          val delta1 = { (outputDelta: INDArray) =>
            -(outputDelta * data0).sumT / (data1 * data1)
          }
          (outputData, delta0, delta1)
        }
      }
    }

    implicit def `Double/INDArray`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Double, Double],
        deepLearning1: DeepLearning.Aux[Operand1, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators./.at[Operand0, Operand1] {
        INDArrayLayer.binary(_, _) { (data0: Double, data1: INDArray) =>
          val outputData = data1 rdiv data0
          val delta0 = { (outputDelta: INDArray) =>
            (outputDelta / data1).sumT
          }
          val delta1 = { (outputDelta: INDArray) =>
            (outputDelta * -data0) / (data1 * data1)
          }
          (outputData, delta0, delta1)
        }
      }
    }

    implicit def `min(INDArray,INDArray)`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
        deepLearning1: DeepLearning.Aux[Operand1, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.min.at[Operand0, Operand1] {
        INDArrayLayer.binary(_, _) { (data0: INDArray, data1: INDArray) =>
          val shape0 = data0.shape
          val shape1 = data1.shape
          val outputShape = autoBroadcastShape(shape0, shape1)
          val broadcastData0 = data0.broadcastFix(outputShape: _*)
          val broadcastData1 = data1.broadcastFix(outputShape: _*)
          val outputData = Transforms.min(broadcastData0, broadcastData1)
          val delta0 = { outputDelta: INDArray =>
            sumAs((broadcastData0 lt broadcastData1) * outputDelta, shape0)
          }
          val delta1 = { outputDelta: INDArray =>
            sumAs((broadcastData0 gt broadcastData1) * outputDelta, shape1)
          }
          (outputData, delta0, delta1)
        }
      }
    }

    implicit def `min(INDArray,Double)`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
        deepLearning1: DeepLearning.Aux[Operand1, Double, Double],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.min.at[Operand0, Operand1] {
        INDArrayLayer.binary(_, _) { (data0: INDArray, data1: Double) =>
          val outputData = Transforms.min(data0, data1)
          val delta0 = { (outputDelta: INDArray) =>
            (data0 lt data1) * outputDelta
          }
          val delta1 = { (outputDelta: INDArray) =>
            ((data0 gt data1) * outputDelta).sumT
          }
          (outputData, delta0, delta1)
        }
      }
    }

    implicit def `min(Double,INDArray)`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Double, Double],
        deepLearning1: DeepLearning.Aux[Operand1, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.min.at[Operand0, Operand1] { (operand0, operand1) =>
        `min(INDArray,Double)`[Operand1, Operand0, Out].apply(operand1, operand0)
      }
    }

    implicit def `max(INDArray,INDArray)`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
        deepLearning1: DeepLearning.Aux[Operand1, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.max.at[Operand0, Operand1] {
        INDArrayLayer.binary(_, _) { (data0: INDArray, data1: INDArray) =>
          val shape0 = data0.shape
          val shape1 = data1.shape
          val outputShape = autoBroadcastShape(shape0, shape1)
          val broadcastData0 = data0.broadcastFix(outputShape: _*)
          val broadcastData1 = data1.broadcastFix(outputShape: _*)
          val outputData = Transforms.max(broadcastData0, broadcastData1)
          val delta0 = { outputDelta: INDArray =>
            sumAs((broadcastData0 gt broadcastData1) * outputDelta, shape0)
          }
          val delta1 = { outputDelta: INDArray =>
            sumAs((broadcastData0 lt broadcastData1) * outputDelta, shape1)
          }
          (outputData, delta0, delta1)
        }
      }
    }

    implicit def `max(INDArray,Double)`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
        deepLearning1: DeepLearning.Aux[Operand1, Double, Double],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.max.at[Operand0, Operand1] {
        INDArrayLayer.binary(_, _) { (data0: INDArray, data1: Double) =>
          val outputData = Transforms.max(data0, data1)
          val delta0 = { (outputDelta: INDArray) =>
            (data0 gt data1) * outputDelta
          }
          val delta1 = { (outputDelta: INDArray) =>
            ((data0 lt data1) * outputDelta).sumT
          }
          (outputData, delta0, delta1)
        }
      }
    }

    implicit def `max(Double,INDArray)`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Double, Double],
        deepLearning1: DeepLearning.Aux[Operand1, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.max.at[Operand0, Operand1] { (operand0, operand1) =>
        `max(INDArray,Double)`[Operand1, Operand0, Out].apply(operand1, operand0)
      }
    }

    implicit def `pow(INDArray,Double)`[Operand0, Operand1, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
        deepLearning1: DeepLearning.Aux[Operand1, Double, Double],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.pow.at[Operand0, Operand1] {
        INDArrayLayer.binary(_, _) { (data0: INDArray, data1: Double) =>
          val outputData = Transforms.pow(data0, data1)
          val delta0 = { (outputDelta: INDArray) =>
            outputDelta * data1 * Transforms.pow(data0, data1 - 1)
          }
          val delta1 = { (outputDelta: INDArray) =>
            (outputDelta * Transforms.log(data0) * outputData).sumT
          }
          (outputData, delta0, delta1)
        }
      }
    }

    implicit def `log(INDArray)`[Operand0, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.log.at[Operand0] {
        INDArrayLayer.unary(_) { data0: INDArray =>
          val outputData = Transforms.log(data0)
          val delta0 = { outputDelta: INDArray =>
            outputDelta / data0
          }
          (outputData, delta0)
        }
      }
    }

    implicit def `exp(INDArray)`[Operand0, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.exp.at[Operand0] {
        INDArrayLayer.unary(_) { data0: INDArray =>
          val outputData = Transforms.exp(data0)
          val delta0 = { outputDelta: INDArray =>
            outputData * outputDelta
          }
          (outputData, delta0)
        }
      }
    }

    implicit def `abs(INDArray)`[Operand0, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.abs.at[Operand0] {
        INDArrayLayer.unary(_) { data0: INDArray =>
          val outputData = Transforms.abs(data0)
          val delta0 = { outputDelta: INDArray =>
            outputDelta * Transforms.sign(data0)
          }
          (outputData, delta0)
        }
      }
    }

    implicit def `sqrt(INDArray)`[Operand0, Out <: INDArrayLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, INDArray, INDArray],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]) = {
      Operators.sqrt.at[Operand0] {
        INDArrayLayer.unary(_) { data0: INDArray =>
          val outputData = Transforms.sqrt(data0)
          val delta0 = { outputDelta: INDArray =>
            outputData * 0.5 / outputData
          }
          (outputData, delta0)
        }
      }
    }

  }

  override type Implicits <: ImplicitsApi

  /** @template */
  type INDArrayLayer <: INDArrayLayerApi with Layer

  @inject
  protected val indArrayLayerFactory: Factory[INDArrayLayer]

  @inject
  protected def indArrayRawForwardParameter: Do[Tape[INDArray, INDArray]] <:< indArrayPartialApplyRawForward.Parameter

  @inject
  protected val indArrayPartialApplyRawForward: PartialApply[indArrayLayerFactory.Constructor,
                                                             shapeless.Witness.`"rawForward"`.T]

  trait INDArrayLayerApi extends super.LayerApi {
    override type Data = INDArray
    override type Delta = INDArray

    /** The original forward operation passed in [[INDArrayLayer$ FloatLayer.apply]].
      *
      * @note This [[rawForward]] may be different from [[forward]],
      *       in the case of [[forward]] was overriden by other plugins, e.g. [[CumulativeINDArrayLayers]].
      */
    protected val rawForward: Do[Tape[INDArray, INDArray]]

    override def forward: Do[Tape[INDArray, INDArray]] = rawForward
  }
  object INDArrayLayer {

    /** @usecase def apply(forward: Do[Tape[INDArray, INDArray]]): INDArrayLayer = ???
      *
      *          Returns a [[INDArrayLayer]] according to the given `forward` operation.
      */
    def apply[Out <: INDArrayLayer](forward: Do[Tape[INDArray, INDArray]])(
        implicit layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]): Out = {
      layerImplicits(
        indArrayPartialApplyRawForward(indArrayLayerFactory.newInstance, indArrayRawForwardParameter(forward)))
    }

    /** Internal helper to create unary [[INDArrayLayer]]. */
    def unary[Operand0, Input0Data, Input0Delta, Out <: INDArrayLayer](
        operand0: Operand0
    )(f: Input0Data => (INDArray, INDArray => Input0Delta))(
        implicit deepLearning0: DeepLearning.Aux[Operand0, Input0Data, Input0Delta],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]
    ): Out = {
      INDArrayLayer(Do.jump().flatMap { _ =>
        deepLearning0.forward(operand0).map {
          case Tape(data0, backward0) =>
            val (outputData, delta0) = f(data0)
            val outputShape = outputData.shape
            def backward(doOutputDelta: Do[INDArray]) = {
              backward0(Do.jump().flatMap { _ =>
                doOutputDelta.map { outputDelta =>
                  delta0(outputDelta.broadcastFix(outputShape: _*))
                }
              })
            }
            Tape(outputData, backward)
        }
      })
    }

    /** Internal helper to create binary [[INDArrayLayer]]. */
    def binary[Operand0, Operand1, Input0Data, Input0Delta, Input1Data, Input1Delta, Out <: INDArrayLayer](
        operand0: Operand0,
        operand1: Operand1
    )(f: (Input0Data, Input1Data) => (INDArray, INDArray => Input0Delta, INDArray => Input1Delta))(
        implicit deepLearning0: DeepLearning.Aux[Operand0, Input0Data, Input0Delta],
        deepLearning1: DeepLearning.Aux[Operand1, Input1Data, Input1Delta],
        layerImplicits: ImplicitApply.Aux[indArrayPartialApplyRawForward.Rest, Out]
    ): Out = {
      INDArrayLayer(Do.jump().flatMap { _ =>
        parallelApply2(deepLearning0.forward(operand0), deepLearning1.forward(operand1)) {
          case (Tape(data0, backward0), Tape(data1, backward1)) =>
            val (outputData, delta0, delta1) = f(data0, data1)
            val outputShape = outputData.shape
            def backward(doOutputDelta: Do[INDArray]) = {
              backward0(Do.jump().flatMap { _ =>
                doOutputDelta.map { outputDelta =>
                  delta0(outputDelta.broadcastFix(outputShape: _*))
                }
              }) |+| backward1(Do.jump().flatMap { _ =>
                doOutputDelta.map { outputDelta =>
                  delta1(outputDelta.broadcastFix(outputShape: _*))
                }
              })
            }
            Tape(outputData, backward)
        }
      })
    }

  }

}
