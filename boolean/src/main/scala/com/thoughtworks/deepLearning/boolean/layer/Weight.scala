package com.thoughtworks.deepLearning
package boolean.layer

import cats._
import com.thoughtworks.deepLearning.boolean.utilities.BooleanMonoidBatch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Weight[Input0 <: Batch](var rawValue: scala.Boolean)
    extends Layer
    with BooleanMonoidBatch
    with BatchId {
  override type Input = Input0
  override type Output = Weight[Input0]
  override type Open = Output

  override def forward(any: BatchId.Aux[Input]) = this

  override def backward(delta: Delta): Unit = {
    rawValue ^= delta.value
  }

  override def value = Eval.now(rawValue)

  override def close(): Unit = {}

  override def open(): Weight[Input0] = this

}
