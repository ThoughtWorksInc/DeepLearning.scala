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
import com.thoughtworks.future.Future
object Layers {
  trait ToLayer[Data, Delta] {
    type OutputLayer

    def toLayer(forward: Do[Tape[Data, Delta]]): OutputLayer

  }

  object ToLayer {
    type Aux[Data, Delta, OutputLayer0] = ToLayer[Data, Delta] {
      type OutputLayer = OutputLayer0
    }
  }

}

/** A plugin that enables [[Layer]] in neural networks. */
trait Layers extends Differentiables {
  trait LayerApi extends DifferentiableApi {
    type Data
    type Delta

    def forward: Do[Tape[Data, Delta]]

  }

  /** A differentiable operation.
    * @template
    */
  type Layer <: LayerApi with Differentiable

  trait ImplicitsApi {
    implicit def layerDeepLearning[From, Data0, Delta0](implicit asLayer: From <:< LayerApi {
      type Data = Data0
      type Delta = Delta0
    }): DeepLearning.Aux[From, Data0, Delta0] = {
      new DeepLearning[From] {
        type Data = Data0
        type Delta = Delta0
        override def forward(from: From): Do[Tape[Data0, Delta0]] = {
          from.forward
        }
      }
    }
  }

  /** @template */
  type Implicits <: ImplicitsApi

}
