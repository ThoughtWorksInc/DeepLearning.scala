package com.thoughtworks.deeplearning
package plugins
import java.util.logging.Logger

import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply, The}
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import com.thoughtworks.raii.shared._
import shapeless.{Poly1, Poly2}
import shapeless.poly.Case1

import scalaz.syntax.all._
import scala.annotation.meta.getter
import scalaz.concurrent.Task

trait Layers {
  trait LayerApi {
    type Data
    type Delta

    def forward: Do[Tape[Data, Delta]]

    protected def handleException(thrown: Throwable): Unit = ()

  }
  type Layer <: LayerApi
  object Layer {
    type Aux[Data0, Delta0] = Layer {
      type Data = Data0
      type Delta = Delta0
    }
  }

  trait ImplicitsApi {
    implicit def layerDeepLearning[From, Data0, Delta0](
        implicit asLayer: From <:< Layer.Aux[Data0, Delta0]): DeepLearning.Aux[From, Data0, Delta0] = {
      new DeepLearning[From] {
        type Data = Data0
        type Delta = Delta0
        override def forward(from: From): Do[Tape[Data0, Delta0]] = {
          from.forward
        }
      }
    }
  }

  type Implicits <: ImplicitsApi

}
