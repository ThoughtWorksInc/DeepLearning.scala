package com.thoughtworks.deeplearning
package plugins
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.feature.ImplicitApply
import com.thoughtworks.raii.asynchronous._
import com.thoughtworks.raii.covariant.{Resource, ResourceT}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.nd4s.IndexRange
import com.thoughtworks.future._
import com.thoughtworks.continuation._
import com.thoughtworks.tryt.covariant.TryT

import scala.util.{Failure, Success, Try}
import scalaz.{-\/, \/-}
import scalaz.syntax.all._

private object CumulativeINDArrayLayers {
  private val Zero = Nd4j.zeros(1, 1)
}

/** A plugin that provides differentiable operators
  * on neural networks whose [[DeepLearning.Data Data]] and [[DeepLearning.Delta Delta]] is [[org.nd4j.linalg.api.ndarray.INDArray]].
  *
  * @note Unlike [[INDArrayLayers]], [[INDArrayLayer]] in this `CumulativeINDArrayLayers` will share [[DeepLearning.Tape Tape]]s
  *       created in [[INDArrayLayer.forward forward]] pass for all dependencies, avoiding re-evaluation
  *       in the case of diamond dependencies in a neural network.
  *
  * @author 杨博 (Yang Bo)
  */
trait CumulativeINDArrayLayers extends INDArrayLayers {
  import CumulativeINDArrayLayers._
  import INDArrayLayers._

  trait INDArrayLayerApi extends super[INDArrayLayers].INDArrayLayerApi {

    private final class Accumulator(val data: INDArray, flushBackward: Do[INDArray] => UnitContinuation[Unit])
        extends Resource[UnitContinuation, Try[Accumulator]] {
      @volatile
      var currentDelta: INDArray = CumulativeINDArrayLayers.Zero

      override def value: Try[Accumulator] = Success(this)

      override def release: UnitContinuation[Unit] = {
        val deltaContinuation: UnitContinuation[INDArray] = Continuation.delay {
          synchronized {
            val delta = currentDelta
            currentDelta = null
            delta
          }
        }

        deltaContinuation.flatMap {
          case Zero =>
            Continuation.now(())
          case nonZeroDelta =>
            flushBackward(Do.now(nonZeroDelta))
        }
      }
    }

    private lazy val sharedAccumulator = {
      val doAccumulator = super.forward.flatMap {
        case Tape(data, flushBackward) =>
          Do(TryT(ResourceT(Continuation.delay[Unit, Resource[UnitContinuation, Try[Accumulator]]] {
            new Accumulator(data, flushBackward)
          })))
      }
      doAccumulator.shared
    }

    /** @usecase def apply(indices: Int*): DoubleLayer = ???
      *
      * @example Given an [[INDArrayLayer]],
      *
      *          {{{
      *          import org.nd4j.linalg.factory.Nd4j
      *          import com.thoughtworks.feature.Factory
      *          import com.thoughtworks.deeplearning.plugins._
      *          import com.thoughtworks.feature.mixins.ImplicitsSingleton
      *          val hyperparameters = Factory[DoubleTraining with CumulativeINDArrayLayers with INDArrayWeights with ImplicitsSingleton with Operators].newInstance()
      *          import hyperparameters.implicits._
      *          val weight = hyperparameters.INDArrayWeight(Nd4j.ones(2, 3))
      *          val indArrayLayer = hyperparameters.INDArrayLayer(weight.forward)
      *          }}}
      *
      *          and select one element in the [[INDArrayLayer]],
      *
      *          {{{
      *          val doubleLayer: hyperparameters.DoubleLayer = indArrayLayer(0, 2)
      *          }}}
      *
      *          when training the selected element,
      *
      *          then the data of the element should be 1.0,
      *          in the original weight,
      *          only the element corresponding to the index get trained.
      *
      *          {{{
      *          doubleLayer.train.map { output =>
      *            output should be(1.0)
      *
      *            import org.nd4s.Implicits._
      *            weight.data(0, 0) should be(1.0)
      *            weight.data(0, 1) should be(1.0)
      *            weight.data(0, 2) should be < 1.0
      *            weight.data(1, 0) should be(1.0)
      *            weight.data(1, 1) should be(1.0)
      *            weight.data(1, 2) should be(1.0)
      *          }
      *          }}}
      *
      */
    def apply[Out <: DoubleLayer](indices: Int*)(
        implicit implicitApply: ImplicitApply.Aux[doublePartialApplyRawForward.Rest, Out]): Out = {
      val doDoubleTape = sharedAccumulator.map { accumulator =>
        def cumulativeBackward(doDelta: Do[Double]): UnitContinuation[Unit] = {
          val Future(TryT(continuation)) = doDelta.run.map { delta: Double =>
            accumulator.synchronized {
              accumulator.currentDelta = accumulator.currentDelta match {
                case null =>
                  throw new IllegalStateException("Cannot perform Tape.backward after the Tape is released")
                case Zero =>
                  val zeros = Nd4j.zeros(accumulator.data.shape(): _*)
                  val indexRanges = indices.map[IndexRange, Array[IndexRange]](i => i)(collection.breakOut)
                  zeros(indexRanges) = delta
                case nonZeroDelta =>
                  val broadcasted = nonZeroDelta.broadcastFix(accumulator.data.shape(): _*)
                  val oldDelta = broadcasted(indices: _*)
                  val indexRanges = indices.map[IndexRange, Array[IndexRange]](i => i)(collection.breakOut)
                  broadcasted(indexRanges) = oldDelta + delta
                  broadcasted
              }
            }
          }

          continuation.map {
            case Success(()) => // Success. Do nothing
            case Failure(e)  => handleException(e)
          }
        }
        Tape(accumulator.data(indices: _*), cumulativeBackward)
      }
      DoubleLayer(doDoubleTape)
    }

    abstract override def forward: Do[Tape[INDArray, INDArray]] = {
      sharedAccumulator.map { accumulator =>
        def cumulativeBackward(doDelta: Do[INDArray]): UnitContinuation[Unit] = {
          val Future(TryT(continuation)) = doDelta.run.map { delta =>
            accumulator.synchronized {
              accumulator.currentDelta = accumulator.currentDelta match {
                case null =>
                  throw new IllegalStateException("Cannot perform Tape.backward after the Tape is released")
                case Zero => delta
                case nonZeroDelta =>
                  def autoBroadcastShape(shape1: Array[Int], shape2: Array[Int]): Array[Int] = {
                    require(shape1.length == shape2.length)
                    shape1.zip(shape2).map {
                      case (1, bSize)                       => bSize
                      case (aSize, 1)                       => aSize
                      case (aSize, bSize) if aSize == bSize => aSize
                    }
                  }

                  val shape = autoBroadcastShape(nonZeroDelta.shape(), delta.shape())
                  val broadcastDelta = nonZeroDelta.broadcastFix(shape: _*)
                  broadcastDelta += delta.broadcastFix(shape: _*)
                  broadcastDelta
              }
            }
          }

          continuation.map {
            case Success(()) => // Success. Do nothing
            case Failure(e)  => handleException(e)
          }
        }
        Tape(accumulator.data, cumulativeBackward)
      }
    }

  }
  override type INDArrayLayer <: INDArrayLayerApi with Layer
}
