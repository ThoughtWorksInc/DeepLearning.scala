package com.thoughtworks.deeplearning.seq.layers

import cats._
import cats.implicits._
import com.thoughtworks.deeplearning._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Get[Input0 <: Batch, ElementData, ElementDelta](
    operand0: Layer.Aux[Input0, Batch.Aux[scala.Seq[ElementData], (Int, ElementDelta)]],
    i: Int)
    extends Layer {

  final class Output private[Get] (upstream: Batch.Aux[scala.Seq[ElementData], (Int, ElementDelta)]) extends Batch {

    type Delta = ElementDelta
    type Data = ElementData
    override def backward(delta: ElementDelta): Unit = {
      upstream.backward((i, delta))
    }

    override def close(): Unit = {
      upstream.close()
    }

    override val value = {
      upstream.value(i)
    }

  }
  override type Input = Input0

  // TODO: Support tail Int
  override def forward(input: BatchId.Aux[Input]) = new BatchId {
    override type Open = Output
    def open() = new Output(operand0.forward(input).open())
  }

}
