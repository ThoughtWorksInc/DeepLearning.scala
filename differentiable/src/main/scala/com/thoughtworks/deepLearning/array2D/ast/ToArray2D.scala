package com.thoughtworks.deepLearning
package array2D.ast

import cats.Eval
import cats.implicits._
import com.thoughtworks.deepLearning.DifferentiableFunction.Ast
import com.thoughtworks.deepLearning.Differentiable.Batch
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning.array2D.utilities._
import com.thoughtworks.deepLearning.array2D.utilities._

final case class FromAstVector[Input0 <: Differentiable](
    operands: Vector[Vector[DifferentiableFunction.Ast[Input0, Differentiable.Batch[Eval[Double], Eval[Double]]]]])
    extends DifferentiableFunction {

  type Input = Input0

  final class Output private[FromAstVector] (upstreams: Vector[Vector[Differentiable.Batch[Eval[Double], Eval[Double]]]])
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

  override def forward(input: Input): Output = {
    new Output(operands.map(_.map(_.forward(input))))
  }
}
