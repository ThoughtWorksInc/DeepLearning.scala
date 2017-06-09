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

/**
  * @example xxx
  *          {{{
  *          import com.thoughtworks.feature.Factory
  *          import com.thoughtworks.deeplearning.plugins._
  *          val hyperparameters = Factory[FloatTraining with FloatLayers with FloatLiterals with ImplicitsSingleton with Operators].newInstance()
  *          import hyperparameters.implicits._
  *          val network: hyperparameters.FloatLayer = (- (6.1f - (- FloatLayerOps(3.4f))))
  *          network.predict
  *          }}}
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait FloatLayers extends RawFloatLayers {

  trait FloatLayerApi extends super[RawFloatLayers].FloatLayerApi {

    private lazy val forward0: Do[Tape[Float, Float]] = {
      val Do(future) = super.forward.flatMap {
        case Tape(data, flushBackward) =>
          Do(Future.delay(new Releasable[Future, Try[Tape[Float, Float]]] {

            @volatile
            private var currentDelta: Float = 0

            override def value: Try[Tape[Float, Float]] = {
              def cumulativeBackward(doDelta: Do[Float]) = {
                Do.run(doDelta)
                  .map { delta =>
                    synchronized {
                      currentDelta += delta
                    }
                  }
                  .get
                  .map {
                    case \/-(()) => // Success. Do nothing
                    case -\/(e) => handleException(e)
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
      val ResourceT(sharedFuture) = ResourceT(future).shared
      Do(sharedFuture)
    }
    abstract override def forward: Do[DeepLearning.Tape[Float, Float]] = forward0

  }
  override type FloatLayer <: FloatLayerApi with Layer
}
