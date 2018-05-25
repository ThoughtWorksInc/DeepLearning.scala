package com.thoughtworks.deeplearning
package plugins
import java.util.logging.Logger

import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.dsl.Dsl.Keyword
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply, The}
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import com.thoughtworks.raii.shared._

import scalaz.syntax.all._
import scala.language.implicitConversions

import scala.annotation.meta.getter
import com.thoughtworks.future.Future
object Layers {
  final case class Eager[Operand0, Data, Delta](operand0: Operand0)(
      implicit val deepLearning: DeepLearning.Aux[Operand0, Data, Delta])
      extends Keyword[Eager[Operand0, Data, Delta], Data]

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
  import com.thoughtworks.deeplearning.plugins.Layers._
  trait LayerApi extends DifferentiableApi {
    type Data
    type Delta

    /** The original forward operation passed in, for creating this [[Layer]].
      *
      * @note This [[rawForward]] may be different from [[forward]],
      *       in the case of [[forward]] was overridden by other plugins, e.g. [[CumulativeFloatLayers]].
      */
    protected val rawForward: Do[Tape[Data, Delta]]

    def forward: Do[Tape[Data, Delta]] = rawForward

  }

  /** A differentiable operation.
    * @template
    */
  type Layer <: LayerApi with Differentiable

  trait ImplicitsApi {

    @inline
    implicit def implicitEager[Operand0](a: Operand0)(
        implicit deepLearning: DeepLearning[Operand0]): Eager[Operand0, deepLearning.Data, deepLearning.Delta] = {
      new Eager[Operand0, deepLearning.Data, deepLearning.Delta](a)(deepLearning)
    }

    implicit def layerDeepLearning[From, Data0, Delta0](implicit asLayer: From <:< LayerApi {
      type Data = Data0
      type Delta = Delta0
    }): DeepLearning.Aux[From, Data0, Delta0] = {
      new DeepLearning[From] {
        type Data = Data0
        type Delta = Delta0
        override def forward(from: From): Do[Tape[Data0, Delta0]] = {
          asLayer(from).forward
        }
      }
    }
  }

  /** @template */
  type Implicits <: ImplicitsApi

}
