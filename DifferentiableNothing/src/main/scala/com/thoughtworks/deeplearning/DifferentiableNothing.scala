package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.Lift._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableNothing {

  /** @template */
  private[deeplearning] type NothingPlaceholder = Placeholder[Nothing, Any]
  val NothingPlaceholder: NothingPlaceholder = implicitly

  object Layers {

    final case class Throw(throwable: () => Throwable) extends Layer with Batch {
      override type Input = Batch
      override type Output = Batch.Aux[Nothing, Any]
      override type Data = Nothing
      override type Delta = Any

      override def forward(input: Input) = this

      override def backward(delta: Delta): Unit = {}

      override def value: Data = {
        throw throwable()
      }

      override def close(): Unit = {}

      override def addReference() = this
    }

  }

  import Layers._

  def `throw`[InputData, InputDelta](throwable: => Throwable)(implicit inputType: Placeholder[InputData, InputDelta])
    : Layer.Aux[Batch.Aux[InputData, InputDelta], NothingPlaceholder.Batch] = {
    Throw(throwable _)
  }

}
