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
import com.thoughtworks.raii.covariant.{Resource, ResourceT}
import com.thoughtworks.tryt.covariant.TryT
import scalaz.{@@, Applicative, Apply}
import scalaz.syntax.all._
import scalaz.std.list._
import scalaz.std.iterable._
import scalaz.std.vector._
import scalaz.Tags.Parallel

import scala.collection.mutable
import scala.util.{Success, Try}

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

  private lazy val doZero: Do[Tensor] = Do.now(Tensor.scalar(0.0f))

  private lazy val doOne: Do[Tensor] = Do.now(Tensor.scalar(1.0f))

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
        Do.garbageCollected(tape.backward(doOne)).map { _: Unit =>
          tape.data
        }
      }
    }

    final def predict: Do[Data] = {
      forward.map(_.data)
    }

  }

  type DeepLearningTensor[Differentiable] = DeepLearning.Aux[Differentiable, Tensor, Tensor]

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

    /** Returns a [[TensorLayer]] according to the given `forward` operation.
      *
      * @usecase def apply(forward: Do[Tape[Tensor, Tensor]]): TensorLayer = ???
      */
    def apply[Out <: TensorLayer](forward: Do[Tape[Tensor, Tensor]])(
        implicit debuggingInformation: TensorLayerDebugging[Out]): Out = {
      debuggingInformation(
        tensorPartialApplyRawForward(tensorLayerFactory.newInstance, tensorRawForwardParameter(forward)))
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

    require(shape.length <= tensor.shape.length, errorMessage)

    def errorMessage = s"Cannot sum [${tensor.shape.mkString(",")}] to [${shape.mkString(",")}]"

    def loop(tensor: Tensor, i: Int, j: Int): Tensor = {
      if (j < tensor.shape.length) {
        if (i >= shape.length) {
          tensor
            .split(j)
            .map { subtensor =>
              loop(subtensor, i + 1, j)
            }
            .reduce(_ + _)
        } else if (tensor.shape(i) != 1 && shape(i) == 1) {
          tensor
            .split(j)
            .map { subtensor =>
              subtensor.permute(
                ((0.until(j).view :+ subtensor.shape.length) ++ j.until(subtensor.shape.length)).toArray)

              loop(subtensor, i + 1, j + 1)
            }
            .reduce(_ + _)
        } else if (tensor.shape(j) == shape(i)) {
          loop(tensor, i + 1, j + 1)
        } else {
          throw new IllegalArgumentException(errorMessage)
        }
      } else {
        tensor
      }
    }

    loop(tensor, 0, 0).ensuring{ t: Tensor=>
      (java.util.Arrays.equals(t.shape, shape))
    }
  }

  def join[Operand0, Out <: TensorLayer](operands: Seq[Operand0], dimension: Int)(
      implicit deepLearning0: DeepLearning.Aux[Operand0, Tensor, Tensor],
      implicitApply: ImplicitApply.Aux[tensorPartialApplyRawForward.Rest, Out]): Out = {
    val doTapes: Vector[ParallelDo[Tape[Tensor, Tensor]]] = operands.map { operand =>
      Parallel(operand.forward)
    }(collection.breakOut(Vector.canBuildFrom))
    TensorLayer(
      Parallel
        .unwrap(Applicative[ParallelDo].sequence(doTapes))
        .map { tapes =>
          val outputData = Tensor.join(tapes.map(_.data), dimension)
          val outputShape = outputData.shape
          def backward(doOutputDelta: Do[Tensor]): UnitContinuation[Unit] = {
            val tapeView: Iterable[Tape[Tensor, Tensor]] = tapes.view
            Parallel.unwrap(tapeView.zipWithIndex.traverse_[ParallelContinuation] {
              case (tape, i) =>
                Parallel(tape.backward(doOutputDelta.map { outputDelta =>
                  outputDelta.broadcast(outputShape).split(dimension).apply(i)
                }))
            })
          }
          Tape(outputData, backward)
        }
    )
  }

  def join[Operand0, Out <: TensorLayer](operands: Seq[Operand0])(
      implicit deepLearning0: DeepLearning.Aux[Operand0, Tensor, Tensor],
      implicitApply: ImplicitApply.Aux[tensorPartialApplyRawForward.Rest, Out]): Out = {
    val doTapes: Vector[ParallelDo[Tape[Tensor, Tensor]]] = operands.map { operand =>
      Parallel(operand.forward)
    }(collection.breakOut(Vector.canBuildFrom))
    TensorLayer(
      Parallel
        .unwrap(Applicative[ParallelDo].sequence(doTapes))
        .map { tapes =>
          val outputData = Tensor.join(tapes.map(_.data))
          val outputShape = outputData.shape
          def backward(doOutputDelta: Do[Tensor]): UnitContinuation[Unit] = {
            val tapeView: Iterable[Tape[Tensor, Tensor]] = tapes.view
            Parallel.unwrap(tapeView.zipWithIndex.traverse_[ParallelContinuation] {
              case (tape, i) =>
                Parallel(tape.backward(doOutputDelta.map { outputDelta =>
                  val element = outputDelta.broadcast(outputShape).split(outputShape.length - 1).apply(i)
                  element
                }))
            })
          }
          Tape(outputData, backward)
        }
    )
  }

  trait ImplicitsApi extends super[Layers].ImplicitsApi {

    /** An implicit wrapper that adds extension methods for differentiable n-dimensional array types
      * that support the [[DeepLearning]] type class.
      */
    implicit final class TensorLayerOps[Operand0](operand0: Operand0)(
        implicit deepLearning0: DeepLearning.Aux[Operand0, Tensor, Tensor]) {

      def translate[Out <: TensorLayer](offset: Array[Double])(
          implicit layerImplicits: ImplicitApply.Aux[tensorPartialApplyRawForward.Rest, Out]): Out = {
        TensorLayer(
          operand0.forward.map {
            case Tape(data0, backwoard0) =>
              val outputData = data0.translate(offset)
              val shape0 = data0.shape
              def backward(doOutputDelta: Do[Tensor]) = {
                backwoard0(doOutputDelta.map(_.broadcast(shape0).translate(offset.map(-_))))
              }
              Tape(outputData, backward)
          }
        )
      }

      def split[Out <: TensorLayer](dimension: Int)(
          implicit layerImplicits: ImplicitApply.Aux[tensorPartialApplyRawForward.Rest, Out]): Do[Seq[Out]] = {

        operand0.forward.flatMap {
          case Tape(inputData, flushBackward) =>
            val resourceContinuation: UnitContinuation[Resource[UnitContinuation, Try[Seq[Out]]]] = {
              UnitContinuation.delay {

                val slices: IndexedSeq[Tensor] = inputData.split(dimension)

                val deltaSlices = mutable.ArraySeq.fill[Do[Tensor]](slices.length)(doZero)

                def release: UnitContinuation[Unit] = UnitContinuation.suspend {
                  flushBackward(deltaSlices.synchronized {
                    val doInputDelta = deltaSlices.toList.sequence.map { tensorSlices =>
                      Tensor.join(tensorSlices, dimension)
                    }
                    for (i <- deltaSlices.indices) {
                      deltaSlices(i) = doZero
                    }
                    doInputDelta
                  })

                }

                def sliceLayers = mutable.ArraySeq.tabulate(slices.length) { i =>
                  val slice = slices(i)
                  val sliceShape = slice.shape

                  def sliceBackward(outputDelta: Do[Tensor]): UnitContinuation[Unit] = UnitContinuation.delay {
                    deltaSlices.synchronized {
                      deltaSlices(i) match {
                        case null =>
                          deltaSlices(i) = outputDelta.map(_.broadcast(sliceShape))
                        case nonNull =>
                          deltaSlices(i) = parallelApply2(nonNull, outputDelta)(_ + _)
                      }
                    }
                  }

                  TensorLayer(Do.now(Tape(slice, sliceBackward)))
                }
                Resource[UnitContinuation, Try[Seq[Out]]](Success(sliceLayers), release)
              }
            }
            Do(TryT(ResourceT(resourceContinuation))).shared
        }
      }

    }

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

    implicit def `Tensor*Tensor`[Operand0, Operand1, Out <: TensorLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Tensor, Tensor],
        deepLearning1: DeepLearning.Aux[Operand1, Tensor, Tensor],
        layerImplicits: ImplicitApply.Aux[tensorPartialApplyRawForward.Rest, Out])
      : Operators.*.Case.Aux[Operand0, Operand1, Out] = {

      Operators.*.at[Operand0, Operand1] { (operand0: Operand0, operand1: Operand1) =>
        TensorLayer(parallelApply2(operand0.forward, operand1.forward) {
          case (Tape(data0, backward0), Tape(data1, backward1)) =>
            val shape0 = data0.shape
            val shape1 = data1.shape
            val outputData = data0 * data1
            def delta0(outputDelta: Tensor) = {
              broadcastThenSum(outputDelta * data1, shape0)
            }
            def delta1(outputDelta: Tensor) = {
              broadcastThenSum(outputDelta * data0, shape1)
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
