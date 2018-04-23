package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.compute.{OpenCL, Tensors}
import com.thoughtworks.continuation._
import com.thoughtworks.future._
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply}
import com.thoughtworks.raii.asynchronous._
import com.thoughtworks.raii.covariant.{MonadicCloseable, Resource}
import shapeless.Witness
import scalaz.syntax.all._

/**
  * @author 杨博 (Yang Bo)
  */
trait TensorWeights extends Tensors with Weights {

  trait TensorWeightApi extends WeightApi with MonadicCloseable[UnitContinuation] {
    this: TensorWeight =>

    override type Delta = Tensor
    override type Data = Tensor // TODO: Data is actually a CachedTensor

    override protected type PartiallyAppliedOptimizer = tensorPartialApplyOriginalDelta.Rest

    override protected def backward[SubtypeOfOptimizer](originalDelta: Do[Tensor])(
        implicit implicitApplyRest: ImplicitApply.Aux[PartiallyAppliedOptimizer, SubtypeOfOptimizer],
        asOptimizer: SubtypeOfOptimizer <:< OptimizerApi { type Delta <: Tensor }): Do[Unit] = {

      val optimizer: OptimizerApi { type Delta <: Tensor } = {
        asOptimizer(
          implicitApplyRest(
            tensorPartialApplyOriginalDelta(tensorPartialApplyWeight(tensorOptimizerFactory.newInstance,
                                                                     tensorWeightParameter(this)),
                                            tensorOriginalDeltaParameter(originalDelta))))
      }
      val doDelta = optimizer.delta
      doDelta.intransitiveFlatMap { delta =>
        Do.garbageCollected((data - delta).doCache.acquire).intransitiveFlatMap {
          case Resource(newData, releaseNewData) =>
            Do.garbageCollected(monadicClose).intransitiveMap { _: Unit =>
              synchronized {
                data = newData
                monadicClose = releaseNewData
              }
            }
        }
      }
    }

    var monadicClose: UnitContinuation[Unit]
  }

  /** @template */
  type TensorWeight <: TensorWeightApi with Weight

  @inject
  protected val tensorWeightFactory: Factory[TensorWeight]

  @inject
  protected val tensorPartialApplyData: PartialApply[tensorWeightFactory.Constructor, Witness.`"data"`.T]

  @inject
  protected val tensorPartialApplyMonadicClose: PartialApply[tensorPartialApplyData.Rest, Witness.`"monadicClose"`.T]

  @inject
  protected def tensorDataParameter: CachedTensor <:< tensorPartialApplyData.Parameter

  @inject
  protected def tensorMonadicCloseParameter: UnitContinuation[Unit] <:< tensorPartialApplyMonadicClose.Parameter

  object TensorWeight {

    /** Returns a RAII managed [[TensorWeight]] with an associated data buffer.
      *
      * @usecase def allocate(data: Tensor): Do[TensorWeight] = ???
      *
      * @example [[TensorWeight.data]] is accessible after the buffer is allocated.
      *
      *          {{{
      *          import com.thoughtworks.feature.mixins.ImplicitsSingleton
      *          import com.thoughtworks.future._
      *          import com.thoughtworks.raii.asynchronous._
      *          import com.thoughtworks.feature.Factory
      *          import scalaz.syntax.all._
      *          import com.thoughtworks.compute.{OpenCL, Tensors}
      *          import com.thoughtworks.deeplearning.plugins.TensorWeights
      *          import com.typesafe.scalalogging.StrictLogging
      *
      *          Do.monadicCloseable{
      *            Factory[
      *              TensorWeights
      *                 with StrictLogging
      *                 with Tensors.UnsafeMathOptimizations
      *                 with OpenCL.LogContextNotification
      *                 with OpenCL.GlobalExecutionContext
      *                 with OpenCL.CommandQueuePool
      *                 with OpenCL.UseAllCpuDevices
      *                 with OpenCL.DontReleaseEventTooEarly
      *                 with OpenCL.SynchronizedCreatingKernel
      *                 with OpenCL.HandleEventInExecutionContextForIntelAndAMDPlatform
      *                 with Tensors.WangHashingRandomNumberGenerator
      *                 with ImplicitsSingleton
      *            ].newInstance(numberOfCommandQueuesPerDevice = 5)
      *          }.flatMap { hyperparameters =>
      *            import hyperparameters._, implicits._
      *            TensorWeight.allocate(Tensor(Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))).map { weight =>
      *              // FIXME: exception thrown here is ignored
      *              weight.data.toString should be("[[1.0,2.0],[3.0,4.0]]")
      *            }
      *          }.run.toScalaFuture
      *          }}}
      */
    def allocate[SubtypeOfWeight <: TensorWeight](initialValue: Tensor)(
        implicit implicitApplyRest: ImplicitApply.Aux[tensorPartialApplyMonadicClose.Rest, SubtypeOfWeight]
    ): Do[SubtypeOfWeight] = {
      Do.monadicCloseable(initialValue.doCache.acquire.map(TensorWeight(_)))
    }

    /** @usecase def apply(resource: Resource[UnitContinuation, CachedTensor]): TensorWeight = ??? */
    def apply[SubtypeOfWeight <: TensorWeight](resource: Resource[UnitContinuation, CachedTensor])(
        implicit implicitApplyRest: ImplicitApply.Aux[tensorPartialApplyMonadicClose.Rest, SubtypeOfWeight]) = {
      val dataApplied = tensorPartialApplyData(tensorWeightFactory.newInstance, tensorDataParameter(resource.value))
      val monadicCloseApplied =
        tensorPartialApplyMonadicClose(dataApplied, tensorMonadicCloseParameter(resource.release))
      implicitApplyRest(monadicCloseApplied)
    }
  }

  trait TensorOptimizerApi extends OptimizerApi { this: TensorOptimizer =>

    type Delta = Tensor

    val weight: TensorWeight

  }

  /** @template */
  type TensorOptimizer <: TensorOptimizerApi with Optimizer

  @inject
  protected val tensorOptimizerFactory: Factory[TensorOptimizer]

  @inject
  protected val tensorPartialApplyWeight: PartialApply[tensorOptimizerFactory.Constructor, Witness.`"weight"`.T]

  @inject
  protected def tensorWeightParameter: TensorWeight <:< tensorPartialApplyWeight.Parameter

  @inject
  protected val tensorPartialApplyOriginalDelta: PartialApply[tensorPartialApplyWeight.Rest,
                                                              Witness.`"originalDelta"`.T]

  @inject
  protected def tensorOriginalDeltaParameter: Do[Tensor] <:< tensorPartialApplyOriginalDelta.Parameter

}
