package com.thoughtworks.deepLearning.seq2D.ast

import cats._
import com.thoughtworks.deepLearning.NeuralNetwork._
import com.thoughtworks.deepLearning.Batch._
import com.thoughtworks.deepLearning.{NeuralNetwork, Batch}
import com.thoughtworks.deepLearning.array2D.utilities._
import com.thoughtworks.deepLearning.double.utilities.DoubleMonoidBatch
import com.thoughtworks.deepLearning.seq2D.utilities.Seq2D

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Get[Input0 <: Batch](operand0: NeuralNetwork.Aux[Input0, Seq2D#ConcreteBatch], i: Int, j: Int) extends NeuralNetwork {
  this: NeuralNetwork.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]] =>

  final class Output private[Get] (upstream: Seq2D#ConcreteBatch) extends DoubleMonoidBatch {

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
  override def forward(input: Input): Output = {
    new Output(operand0.forward(input))
  }

}
