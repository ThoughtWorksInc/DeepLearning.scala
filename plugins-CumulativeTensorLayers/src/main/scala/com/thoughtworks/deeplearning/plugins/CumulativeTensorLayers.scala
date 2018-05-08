package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.feature.ImplicitApply
import com.thoughtworks.raii.asynchronous._
import com.thoughtworks.raii.covariant.{Releasable, Resource, ResourceT}
import com.thoughtworks.future._
import com.thoughtworks.continuation._
import com.thoughtworks.tryt.covariant.TryT

import scala.util.{Failure, Success, Try}
import scalaz.{-\/, \/-}
import scalaz.syntax.all._
import scalaz.Tags.Parallel

/** A plugin that provides differentiable operators
  * on neural networks whose [[DeepLearning.Data Data]] and [[DeepLearning.Delta Delta]] is [[org.nd4j.linalg.api.ndarray.Tensor]].
  *
  * @note Unlike [[TensorLayers]], [[TensorLayer]] in this `CumulativeTensorLayers` will share [[DeepLearning.Tape Tape]]s
  *       created in [[TensorLayer.forward forward]] pass for all dependencies, avoiding re-evaluation
  *       in the case of diamond dependencies in a neural network.
  *
  * @author 杨博 (Yang Bo)
  */
trait CumulativeTensorLayers extends TensorLayers {
  import TensorLayers._
  private val zero = Tensor.scalar(0.0f)
  private val doZero = Do.now(zero)

  trait TensorLayerApi extends super[TensorLayers].TensorLayerApi {

    private final class Accumulator(val data: Tensor, flushBackward: Do[Tensor] => UnitContinuation[Unit])
        extends Resource[UnitContinuation, Try[Accumulator]] {
      @volatile
      var currentDelta: Do[Tensor] = doZero

      override def value: Try[Accumulator] = Success(this)

      override def release: UnitContinuation[Unit] = UnitContinuation.suspend {
        val delta: Do[Tensor] = {
          synchronized {
            val delta = currentDelta
            currentDelta = null.asInstanceOf[Do[Tensor]]
            delta
          }
        }
        if (delta == doZero) {
          Continuation.now(())
        } else {
          flushBackward(delta)
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

    abstract override def forward: Do[Tape[Tensor, Tensor]] = {
      sharedAccumulator.map { accumulator =>
        def cumulativeBackward(doDelta: Do[Tensor]): UnitContinuation[Unit] = UnitContinuation.delay {
          synchronized {
            accumulator.currentDelta = accumulator.currentDelta match {
              case null =>
                throw new IllegalStateException("Cannot perform Tape.backward after the Tape is released")
              case `zero` =>
                doDelta
              case nonZeroAccumulator =>
                Parallel.unwrap(
                  asynchronousDoParallelApplicative.apply2(Parallel(nonZeroAccumulator), Parallel(doDelta))(_ + _))
            }
          }
        }
        Tape(accumulator.data, cumulativeBackward)
      }
    }

  }
  override type TensorLayer <: TensorLayerApi with Layer
}
