package com.thoughtworks.deepLearning
package array2D.layer

import cats.{Applicative, Eval, Traverse}
import cats.implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning.array2D.utilities._

import scala.language.higherKinds

object ToArray2D {

  private[ToArray2D] implicit object SeqInstances extends Traverse[Seq] {
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

final case class ToArray2D[Input0 <: Batch](
    operands: Seq[Seq[Layer.Aux[Input0, Batch.Aux[Eval[Double], Eval[Double]]]]])
    extends Layer {

  import ToArray2D.SeqInstances

  type Input = Input0

  final class Output private[ToArray2D] (upstreams: Seq[Seq[Batch.Aux[Eval[Double], Eval[Double]]]])
      extends Array2DSemigroupBatch {
    override def backward(delta: Eval[INDArray]): Unit = {
      for ((row, i) <- upstreams.view.zipWithIndex; (upstream, j) <- row.zipWithIndex) {
        upstream.backward(delta.map(_(i, j)))
      }

    }

    override val value = {
      upstreams.traverse(_.traverse(_.value)).map(_.toNDArray).memoize
    }

    override def close(): Unit = {
      upstreams.foreach(_.foreach(_.close()))
    }

  }

  override def forward(input: BatchId.Aux[Input]) = new BatchId {
    override type Open = Output
    override def open() = {
      new Output(operands.map(_.map(_.forward(input).open())))
    }
  }
}
