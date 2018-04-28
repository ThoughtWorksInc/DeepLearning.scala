package com.thoughtworks.deeplearning.plugins

/** A plugin that automatically names [[Layer]]s and [[Weight]]s.
  *
  * @author 杨博 (Yang Bo)
  */
trait Names {

  trait DifferentiableApi {
    def fullName: sourcecode.FullName
    def name: sourcecode.Name

    override def toString: String = {
      raw"""Weight[fullName=${fullName.value}]"""
    }

  }
  type Differentiable <: DifferentiableApi

}
