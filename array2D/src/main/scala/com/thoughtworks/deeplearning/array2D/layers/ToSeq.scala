package com.thoughtworks.deeplearning.array2D.layers

import cats._
import cats.implicits._
import org.nd4s.Implicits._
import com.thoughtworks.deeplearning.Batch._
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.array2D.utilities._
import com.thoughtworks.deeplearning.{Batch, BatchId, BufferedLayer, Layer}
import com.thoughtworks.deeplearning.seq2D.utilities.{Seq2D, Seq2DBatch}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class ToSeq[Input0 <: Batch](operand: Layer.Aux[Input0, Array2D#Batch]) extends BufferedLayer {
  override type Input = Input0

  final class BufferedBatch private[ToSeq] (override val input: BatchId.Aux[Input], upstream: Array2D#Batch)
      extends ReferenceCount
      with Seq2DBatch {

    private def zeroDelta =
      upstream.value.map { upstreamData =>
        Nd4j.zeros(upstreamData.shape: _*)
      }.memoize

    @volatile
    var upstreamDelta = zeroDelta

    override protected def flush(): Unit = {
      upstream.backward(synchronized {
        val oldDelta = upstreamDelta
        upstreamDelta = zeroDelta
        oldDelta
      })
    }

    override protected def closeUpstreams(): Unit = {
      upstream.close()
    }

    override def backward(delta: Delta): Unit = {
      synchronized {
        val (i, j, value) = delta.value
        upstreamDelta.value(i, j) = upstreamDelta
            .value(i, j) + value // Cannot use += because of https://issues.scala-lang.org/browse/SI-10021
      }
    }

    override val value: Data = {
      upstream.value.map { ndarray: INDArray =>
        val doubleArray = ndarray.data.asDouble()
        for (i <- (0 until ndarray.rows).view) yield {
          doubleArray.view(i * ndarray.columns, (i + 1) * ndarray.columns)
        }
      }.memoize
    }
  }

  override protected def rawForward(input: BatchId.Aux[Input]): BufferedBatch = {
    new BufferedBatch(input, operand.forward(input).open())
  }
}
