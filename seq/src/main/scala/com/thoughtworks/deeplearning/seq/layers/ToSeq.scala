package com.thoughtworks.deeplearning.seq.layers

import cats._
import cats.implicits._
import com.thoughtworks.deeplearning._

import com.thoughtworks.deeplearning.utilities.CloseableOnce

import scala.language.higherKinds

object ToSeq {

  private[ToSeq] implicit object SeqInstances extends Traverse[Seq] {
    override def traverse[G[_]: Applicative, A, B](fa: Seq[A])(f: (A) => G[B]): G[Seq[B]] = {
      fa.foldRight((Vector.empty[B]: Seq[B]).pure[G]) { (a: A, accumulation: G[Seq[B]]) =>
        f(a).map2(accumulation)(_ +: _)
      }
    }

    override def foldLeft[A, B](fa: Seq[A], b: B)(f: (B, A) => B): B = {
      fa.foldLeft(b)(f)
    }

    override def foldRight[A, B](fa: Seq[A], lb: Eval[B])(f: (A, Eval[B]) => Eval[B]): Eval[B] = {
      fa.foldRight(lb)(f)
    }
  }

}

final case class ToSeq[Input0 <: Batch, ElementData, ElementDelta](
    operands: Seq[Layer.Aux[Input0, Batch.Aux[ElementData, ElementDelta]]])
    extends Layer {

  import ToSeq.SeqInstances

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

  }

  override def forward(input: BatchId.Aux[Input]) = new BatchId {
    override type Open = Output
    override def open() = {
      new Output(operands.map(_.forward(input).open()))
    }
  }
}
