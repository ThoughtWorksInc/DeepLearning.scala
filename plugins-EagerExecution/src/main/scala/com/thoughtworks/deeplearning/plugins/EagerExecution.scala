package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.plugins.EagerExecution.Eager
import com.thoughtworks.dsl.Dsl
import com.thoughtworks.dsl.Dsl.{Keyword, shift}
import com.thoughtworks.feature.Factory

import scala.annotation.compileTimeOnly

/**
  * @author 杨博 (Yang Bo)
  */
trait EagerExecution extends Layers {

//
//  trait LayerApi extends super.LayerApi {
//
//
//    @shift
//    @compileTimeOnly(
//      """This method requires the compiler plugin: `addCompilerPlugin("com.thoughtworks.dsl" %% "compilerplugins-bangnotation" % "latest.release")` and must only be called inside a code block annotated as `@reset`.""")
//    final def data : Data = {
//      throw new IllegalAccessException(
//        """This method requires the compiler plugin: `addCompilerPlugin("com.thoughtworks.dsl" %% "compilerplugins-bangnotation" % "latest.release")` and must only be called inside a code block annotated as `@reset`."""
//      )
//    }
//
//    @inline
//    final def cpsApply[Domain](handler: Data => Domain): Domain = {
//      ???
////      dsl.interpret(this, handler)
//    }
//
//  }
//
//  type Layer <: LayerApi
//  def Eager[A](a: A)(implicit deepLearning: DeepLearning[A]): Eager[A, deepLearning.Data, deepLearning.Delta] = {
//    new Eager[A, deepLearning.Data, deepLearning.Delta](a, deepLearning)
//  }

}

object EagerExecution {
  final case class Eager[Differentiable, Data, Delta](differentiable: Differentiable)(
      implicit
      deepLearning: DeepLearning.Aux[Differentiable, Data, Delta])
      extends Keyword[Eager[Differentiable, Data, Delta], Data]
}
