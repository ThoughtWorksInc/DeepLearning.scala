package com.thoughtworks.deeplearning
package plugins
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import com.thoughtworks.raii.shared._
import com.thoughtworks.raii.covariant.{Releasable, ResourceT}
import com.thoughtworks.tryt.covariant.TryT

import scalaz.concurrent.Future
import scala.util.{Success, Try}
import scalaz.{-\/, \/-}
import scalaz.syntax.all._

/** A plugin that provides differentiable operators
  * on neural networks whose [[DeepLearning.Data Data]] and [[DeepLearning.Delta Delta]] is [[scala.Float]].
  *
  * @note Unlike [[RawFloatLayers]], [[FloatLayer]] in this `FloatLayers` will share the [[DeepLearning.Tape Tape]]
  *       created in [[FloatLayer.forward forward]] pass pass for all dependencies, avoiding re-evaluation
  *       in the case of diamond dependencies in a neural network.
  *
  * @author 杨博 (Yang Bo)
  */
trait FloatLayers extends RawFloatLayers {

  trait FloatLayerApi extends super[RawFloatLayers].FloatLayerApi {

    private def doCumulativeTape: Do[Tape[Float, Float]] = {
      super.forward.flatMap {
        case Tape(data, flushBackward) =>
          Do(Future.delay(new Releasable[Future, Try[Tape[Float, Float]]] {

            @volatile
            private var currentDelta: Float = 0

            override def value: Try[Tape[Float, Float]] = {
              def cumulativeBackward(doDelta: Do[Float]): Future[Unit] = {
                Do.run(doDelta)
                  .map { delta =>
                    synchronized {
                      currentDelta += delta
                    }
                  }
                  .get
                  .map {
                    case \/-(()) => // Success. Do nothing
                    case -\/(e)  => handleException(e)
                  }
              }

              Success(Tape(data, cumulativeBackward))
            }

            override def release(): Future[Unit] = {
              flushBackward(Do.delay {
                synchronized {
                  val delta = currentDelta
                  currentDelta = 0
                  delta
                }
              })
            }

          }))
      }
    }

    @transient
    private lazy val sharedForward: Do[Tape[Float, Float]] = {
      Do.shared(doCumulativeTape)
    }

    abstract override def forward: Do[DeepLearning.Tape[Float, Float]] = sharedForward

  }
  override type FloatLayer <: FloatLayerApi with Layer
}
