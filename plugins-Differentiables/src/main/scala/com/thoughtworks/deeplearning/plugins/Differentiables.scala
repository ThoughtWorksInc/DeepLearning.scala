package com.thoughtworks.deeplearning.plugins

/**
  * @author 杨博 (Yang Bo)
  */
trait Differentiables {

  trait DifferentiableApi {

    protected def handleException(throwable: Throwable): Unit = {
      throwable.printStackTrace()
    }
  }

  type Differentiable <: DifferentiableApi

}
