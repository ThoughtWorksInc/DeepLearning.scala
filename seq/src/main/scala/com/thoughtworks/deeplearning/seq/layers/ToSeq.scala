package com.thoughtworks.deeplearning.seq.layers

import cats._
import cats.implicits._
import com.thoughtworks.deeplearning._

import com.thoughtworks.deeplearning.utilities.CloseableOnce

import scala.language.higherKinds

final case class ToSeq[Input0 <: Batch, ElementData, ElementDelta](
    operands: scala.Seq[Layer.Aux[Input0, Batch.Aux[ElementData, ElementDelta]]])
    extends Layer {

  type Input = Input0

  final class Output private[ToSeq] (upstreams: scala.Seq[Batch.Aux[ElementData, ElementDelta]])
      extends Batch
      with CloseableOnce {

    override type Data = scala.Seq[ElementData]
    override type Delta = (Int, ElementDelta)

    override def backward(pair: (Int, ElementDelta)): Unit = {
      val (i, delta) = pair
      upstreams(i).backward(delta)
    }

    override val value = {
      upstreams.map(_.value)
    }

    override def close(): Unit = {
      super.close()
      upstreams.foreach(_.close())
    }

  }

  override def forward(input: BatchId.Aux[Input]) = new BatchId {
    override type Open = Output
    override def open() = {
      new Output(operands.map(_.forward(input).open()))
    }
  }
}
