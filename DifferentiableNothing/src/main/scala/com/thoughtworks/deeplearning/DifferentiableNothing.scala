package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Tape
import com.thoughtworks.deeplearning.Symbolic._

/**
  * A namespace of common operators for all layers.
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableNothing {

  /** @template */
  private[deeplearning] type NothingPlaceholder = Placeholder[Nothing, Any]
  private[deeplearning] val NothingPlaceholder: NothingPlaceholder = new Placeholder

  object Layers {

    final case class Throw(throwable: () => Throwable) extends Layer with Tape {
      override type Input = Tape
      override type Output = Tape.Aux[Nothing, Any]
      override type Data = Nothing
      override type Delta = Any

      override def forward(input: Input) = this

      override protected def forceBackward(delta: Delta): Unit = {}

      override def value: Data = {
        throw throwable()
      }

      override def close(): Unit = {}

      override def duplicate() = this

      override def isTrainable = false

    }

  }

  import Layers._

  def `throw`[InputData, InputDelta](throwable: => Throwable)(implicit inputType: Placeholder[InputData, InputDelta])
    : Layer.Aux[Tape.Aux[InputData, InputDelta], NothingPlaceholder.Tape] = {
    Throw(throwable _)
  }

}
