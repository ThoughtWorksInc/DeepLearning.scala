package com.thoughtworks.deeplearning.seq.layers

import cats._
import cats.implicits._
import com.thoughtworks.deeplearning._

import com.thoughtworks.deeplearning.utilities.CloseableOnce

import language.higherKinds

final case class ToSeq[Input0 <: Batch, ElementData, ElementDelta](
    operands: Seq[Layer.Aux[Input0, Batch.Aux[ElementData, ElementDelta]]])
    extends Layer {

  type Input = Input0

  final class Output private[ToSeq] (upstreams: Seq[Batch.Aux[ElementData, ElementDelta]])
      extends Batch
      with CloseableOnce {

    override type Data = Seq[ElementData]
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

    override def addReference() = new Output(upstreams.map(_.addReference()))

  }

  override def forward(input: Input) = new Output(operands.map(_.forward(input)))

}
