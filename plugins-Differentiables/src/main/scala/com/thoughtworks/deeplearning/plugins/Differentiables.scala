package com.thoughtworks.deeplearning.plugins
import com.thoughtworks.continuation._
/**
  * @author 杨博 (Yang Bo)
  */
trait Differentiables {

  trait DifferentiableApi {

    protected def handleException(throwable: Throwable): UnitContinuation[Unit] = {
      UnitContinuation.delay {
        throwable.printStackTrace()
      }
    }
  }

  type Differentiable <: DifferentiableApi

}
