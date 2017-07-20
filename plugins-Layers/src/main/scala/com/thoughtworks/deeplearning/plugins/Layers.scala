package com.thoughtworks.deeplearning
package plugins
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.raii.asynchronous.Do

/** A plugin that enables [[Layer]] in neural networks. */
trait Layers {
  trait LayerApi {
    type Data
    type Delta

    def forward: Do[Tape[Data, Delta]]

    protected def handleException(thrown: Throwable): Unit = ()

  }

  /** A differentiable operation.
    * @template
    */
  type Layer <: LayerApi

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
