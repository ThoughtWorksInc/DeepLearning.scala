package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.compute.Tensors
import com.thoughtworks.continuation._
import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply}
import com.thoughtworks.raii.asynchronous._
import com.thoughtworks.future._
import com.thoughtworks.deeplearning.DeepLearning.ops._
import scalaz.{Applicative, Apply}
import scalaz.syntax.all._
import scalaz.Tags.Parallel
private[plugins] object TensorLayers {

  private val MergeUnit = { (_: Unit, _: Unit) =>
    ()
  }

  @inline
  def parallelUnitContinuation(continuation0: UnitContinuation[Unit],
                               continuation1: UnitContinuation[Unit]): UnitContinuation[Unit] = {
    Parallel.unwrap(continuationParallelApplicative.apply2(Parallel(continuation0), Parallel(continuation1))(MergeUnit))
  }

  def parallelApply2[A, B, C](do0: Do[A], do1: Do[B])(f: (A, B) => C): Do[C] = {
    Parallel.unwrap(asynchronousDoParallelApplicative.apply2(Parallel(do0), Parallel(do1))(f))
  }

  @inline
  def parallelTuple2[A, B](do0: Do[A], do1: Do[B]): Do[(A, B)] = {
    Parallel.unwrap(asynchronousDoParallelApplicative.tuple2(Parallel(do0), Parallel(do1)))
  }

}

/**
  * @author 杨博 (Yang Bo)
  */
trait TensorLayers extends Tensors with Layers {
  import TensorLayers._
  private lazy val One: Tensor = Tensor.scalar(1.0f)

  trait TensorLayerApi extends super[Layers].LayerApi {
    type Data = Tensor
    type Delta = Tensor

    /** The original forward operation passed in [[TensorLayer$ TensorLayer.apply]].
      *
      * @note This [[rawForward]] may be different from [[forward]],
      *       in the case of [[forward]] was overriden by other plugins, e.g. [[CumulativeTensorLayers]].
      */
    protected val rawForward: Do[Tape[Tensor, Tensor]]

    override def forward: Do[Tape[Tensor, Tensor]] = rawForward

    final def train: Do[Data] = {
      forward.flatMap[Data] { tape =>
        Do.garbageCollected(tape.backward(Do.now(One))).map { _: Unit =>
          tape.data
        }
      }
    }

    final def predict: Do[Data] = {
      forward.map(_.data)
    }

  }

  type TensorLayer <: TensorLayerApi with Layer

  @inject
  protected val tensorLayerFactory: Factory[TensorLayer]

  @inject
  protected def tensorRawForwardParameter: Do[Tape[Tensor, Tensor]] <:< tensorPartialApplyRawForward.Parameter

  @inject
  protected val tensorPartialApplyRawForward: PartialApply[tensorLayerFactory.Constructor,
                                                           shapeless.Witness.`"rawForward"`.T]

  /** Contains line number, caller and other debugging information for [[TensorLayer]]
    * @documentable
    */
  type TensorLayerDebugging[Out <: TensorLayer] = ImplicitApply.Aux[tensorPartialApplyRawForward.Rest, Out]

  object TensorLayer {

    /** @usecase def apply(forward: Do[Tape[Tensor, Tensor]]): TensorLayer = ???
      *
      *          Returns a [[TensorLayer]] according to the given `forward` operation.
      */
    def apply[Out <: TensorLayer](forward: Do[Tape[Tensor, Tensor]])(
        implicit implicitApply: ImplicitApply.Aux[tensorPartialApplyRawForward.Rest, Out]): Out = {
      implicitApply(tensorPartialApplyRawForward(tensorLayerFactory.newInstance, tensorRawForwardParameter(forward)))
    }
  }

  private def broadcastThenSum(tensor: Tensor, shape: Array[Int]) = {
    val broad = if (tensor.shape.length < shape.length) {
      tensor.broadcast(Array.tabulate(shape.length) { i =>
        if (i < tensor.shape.length) { tensor.shape(i) } else {
          shape(i)
        }
      })
    } else {
      tensor
    }
    sumAs(broad, shape)
  }

  // TODO: Use device side loops once https://github.com/ThoughtWorksInc/Compute.scala/issues/62 is implemented
  private def sumAs(tensor: Tensor, shape: Array[Int]) = {

    require(shape.length == tensor.shape.length, errorMessage)

    def errorMessage = s"Cannot sum [${tensor.shape.mkString(",")}] to [${shape.mkString(",")}]"
    def loop(tensor: Tensor, i: Int): Tensor = {
      if (i < shape.length) {
        if (tensor.shape(i) != 1 && shape(i) == 1) {
          tensor
            .split(i)
            .map { subtensor =>
              loop(subtensor, i + 1)
            }
            .reduce(_ + _)
        } else if (tensor.shape(i) == shape(i)) {
          loop(tensor, i + 1)
        } else {
          throw new IllegalArgumentException(errorMessage)
        }
      } else {
        tensor
      }
    }

    loop(tensor, 0)
  }

  trait ImplicitsApi extends super[Layers].ImplicitsApi {
    implicit def `Tensor+Tensor`[Operand0, Operand1, Out <: TensorLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Tensor, Tensor],
        deepLearning1: DeepLearning.Aux[Operand1, Tensor, Tensor],
        layerImplicits: ImplicitApply.Aux[tensorPartialApplyRawForward.Rest, Out]) = {

      Operators.+.at[Operand0, Operand1] { (operand0: Operand0, operand1: Operand1) =>
        TensorLayer(parallelApply2(operand0.forward, operand1.forward) {
          case (Tape(data0, backward0), Tape(data1, backward1)) =>
            val shape0 = data0.shape
            val shape1 = data1.shape
            val outputData = data0 + data1
            def delta0(outputDelta: Tensor) = {
              broadcastThenSum(outputDelta, shape0)
            }
            def delta1(outputDelta: Tensor) = {
              broadcastThenSum(outputDelta, shape1)
            }
            def backward(outputDelta: Do[Tensor]) = {
              parallelUnitContinuation(backward0(outputDelta.map(delta0)), backward1(outputDelta.map(delta1)))
            }
            Tape(outputData, backward)
        })
      }
    }

  }
  type Implicits <: ImplicitsApi
}
