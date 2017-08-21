package com.thoughtworks.deeplearning
package plugins
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.raii.asynchronous._
import com.thoughtworks.raii.shared._
import com.thoughtworks.raii.covariant.{Resource, ResourceT}
import com.thoughtworks.tryt.covariant.TryT
import com.thoughtworks.continuation._
import com.thoughtworks.future._

import scala.util.{Failure, Success, Try}
import scalaz.syntax.all._

/** A plugin that provides differentiable operators
  * on neural networks whose [[DeepLearning.Data Data]] and [[DeepLearning.Delta Delta]] is [[scala.Float]].
  *
  * @note Unlike [[FloatLayers]], [[FloatLayer]] in this `CumulativeFloatLayers` will share [[DeepLearning.Tape Tape]]s
  *       created in [[FloatLayer.forward forward]] pass pass for all dependencies, avoiding re-evaluation
  *       in the case of diamond dependencies in a neural network.
  *
  * @example Given two [[FloatWeights.FloatWeight FloatWeight]]s,
  *
  *          {{{
  *          import com.thoughtworks.deeplearning.plugins._
  *          import com.thoughtworks.feature.Factory
  *          import com.thoughtworks.feature.mixins.ImplicitsSingleton
  *          val hyperparameters = Factory[FloatTraining with ImplicitsSingleton with Operators with CumulativeFloatLayers with FloatWeights].newInstance()
  *          import hyperparameters.implicits._
  *          val weight1 = hyperparameters.FloatWeight(10)
  *          val weight2 = hyperparameters.FloatWeight(300)
  *          }}}
  *
  *          when adding them together,
  *
  *          {{{
  *          val weight1PlusWeight2 = weight1 + weight2
  *          }}}
  *
  *          then the training result should be applied on both weight
  *
  *          {{{
  *          weight1PlusWeight2.train.map { result =>
  *            result should be(310.0f)
  *
  *            weight2.data should be < 300.0f
  *            weight1.data should be < 10.0f
  *          }
  *          }}}
  *
  * @example Given a [[FloatWeights.FloatWeight FloatWeight]],
  *
  *          {{{
  *          import com.thoughtworks.deeplearning.plugins._
  *          import com.thoughtworks.feature.Factory
  *          import com.thoughtworks.feature.mixins.ImplicitsSingleton
  *          val hyperparameters = Factory[FloatTraining with ImplicitsSingleton with Operators with CumulativeFloatLayers with FloatWeights].newInstance()
  *          import hyperparameters.implicits._
  *          val weight1 = hyperparameters.FloatWeight(10)
  *          }}}
  *
  *          then the training result should be applied on it
  *
  *          {{{
  *          weight1.train.map { result =>
  *            result should be(10.0f)
  *
  *            weight1.data should be < 10.0f
  *          }
  *          }}}
  *
  * @author 杨博 (Yang Bo)
  */
trait CumulativeFloatLayers extends FloatLayers {

  trait FloatLayerApi extends super[FloatLayers].FloatLayerApi {

    @transient
    private lazy val sharedForward: Do[Tape[Float, Float]] = {
      val doCumulativeTape: Do[Tape[Float, Float]] = super.forward.flatMap {
        case Tape(data, flushBackward) =>
          Do(TryT(ResourceT(Continuation.delay[Unit, Resource[UnitContinuation, Try[Tape[Float, Float]]]] {
            new Resource[UnitContinuation, Try[Tape[Float, Float]]] {

              @volatile
              private var currentDelta: Float = 0

              override def value: Try[Tape[Float, Float]] = {
                def cumulativeBackward(doDelta: Do[Float]): UnitContinuation[Unit] = {
                  val Future(TryT(continuation)) = doDelta.run.map { delta =>
                    synchronized {
                      currentDelta += delta
                    }
                  }
                  continuation.map {
                    case Success(()) => // Success. Do nothing
                    case Failure(e)  => handleException(e)
                  }
                }

                Success(Tape(data, cumulativeBackward))
              }

              override def release(): UnitContinuation[Unit] = {
                val deltaContinuation: UnitContinuation[Float] = Continuation.delay {
                  synchronized {
                    val delta = currentDelta
                    currentDelta = 0
                    delta
                  }
                }

                deltaContinuation.flatMap { delta =>
                  if (delta == 0) {
                    Continuation.now(())
                  } else {
                    flushBackward(Do.delay {
                      delta
                    })
                  }
                }
              }

            }
          })))
      }

      doCumulativeTape.shared
    }

    abstract override def forward: Do[DeepLearning.Tape[Float, Float]] = sharedForward

  }
  override type FloatLayer <: FloatLayerApi with Layer
}
