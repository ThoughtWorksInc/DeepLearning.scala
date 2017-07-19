package com.thoughtworks.deeplearning.plugins

/** A plugin that automatically names [[Layer]]s and [[Weight]]s.
  *
  * @author 杨博 (Yang Bo)
  */
trait Naming extends Layers with Weights {

  trait LayerApi extends super.LayerApi { this: Layer =>
    def fullName: sourcecode.FullName
    def name: sourcecode.Name

    override def toString: String = {
      raw"""Layer[fullName=${fullName.value}]"""
    }
  }
  override type Layer <: LayerApi

  trait WeightApi extends super.WeightApi { this: Weight =>
    def fullName: sourcecode.FullName
    def name: sourcecode.Name

    override def toString: String = {
      raw"""Weight[fullName=${fullName.value}]"""
    }

  }
  override type Weight <: WeightApi
  override type Implicits <: super[Layers].ImplicitsApi with super[Weights].ImplicitsApi

}
