package com.thoughtworks.deepLearning.any.ast

import com.thoughtworks.deepLearning.NeuralNetwork._
import com.thoughtworks.deepLearning.Batch._
import com.thoughtworks.deepLearning.{Batch, NeuralNetwork}

/**
  * FIXME: Use smart pointer instead
  * This [[Identity]] must be a singleton to count number of [[forward]] calls for same [[Input]] instance.
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
sealed abstract case class Identity[Input0 <: Batch] private () extends Cached {
  type Input = Input0

  protected final class SharedBatch(override val input: Input0) extends ReferenceCount {
    type Data = input.Data
    type Delta = input.Delta
    override protected def flush(): Unit = {}

    /**
      * Closes upstream batch of this [[SharedBatch]]
      */
    override protected def closeUpstreams(): Unit = {
      input.close()
    }

    override def backward(delta: Delta): Unit = {
      input.backward(delta)
    }

    override def value: Data = input.value
  }

  /**
    * Performs the underlying forward pass.
    *
    * @return a [[Batch]] that will be cached for subsequent [[#forward]]
    */
  override protected def rawForward(input: Input0) = new SharedBatch(input)
}

object Identity {

  private object SingletonInstance extends Identity[Batch]

  def apply[Input <: Batch]() = SingletonInstance.asInstanceOf[Identity[Input]]

}
