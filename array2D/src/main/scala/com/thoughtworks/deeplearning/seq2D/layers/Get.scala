package com.thoughtworks.deeplearning.seq2D.layers

import cats._
import com.thoughtworks.deeplearning.{Batch, BatchId, Layer}
import com.thoughtworks.deeplearning.double.utilities.DoubleMonoidBatch
import com.thoughtworks.deeplearning.seq2D.utilities.Seq2D

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Get[Input0 <: Batch](operand0: Layer.Aux[Input0, Seq2D#Batch], i: Int, j: Int) extends Layer {
  this: Layer.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]] =>

  final class Output private[Get] (upstream: Seq2D#Batch) extends DoubleMonoidBatch {

    override def backward(delta: Eval[scala.Double]): Unit = {
      upstream.backward(delta.map((i, j, _)))
    }

    override def close(): Unit = {
      upstream.close()
    }

    override val value: Eval[scala.Double] = {
      upstream.value.map { v =>
        v(i)(j)
      }.memoize
    }

  }
  override type Input = Input0

  // TODO: Support tail Int
  override def forward(input: BatchId.Aux[Input]) = new BatchId {
    override type Open = Output
    def open() = new Output(operand0.forward(input).open())
  }

}
