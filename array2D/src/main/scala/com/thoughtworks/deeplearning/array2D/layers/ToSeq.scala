package com.thoughtworks.deeplearning.array2D.layers

import cats._
import cats.implicits._
import org.nd4s.Implicits._
import com.thoughtworks.deeplearning.Batch._
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.array2D.utilities._
import com.thoughtworks.deeplearning.{Batch, BufferedLayer, Layer}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

object ToSeq {
  private[ToSeq] trait Seq2DBatch extends Batch {
    override type Data = scala.Seq[scala.Seq[Eval[scala.Double]]]
    override type Delta = (scala.Int, (scala.Int, Eval[scala.Double]))
  }
}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class ToSeq[Input0 <: Batch](operand: Layer.Aux[Input0, Array2D#Batch]) extends BufferedLayer.Unary {
  import ToSeq._
  type BufferedBatch =
    UnaryBatch with Seq2DBatch

  type Input = Input0

  override protected def rawForward(input0: Input): BufferedBatch = {
    new {
      override val input = input0
    } with UnaryBatch with Seq2DBatch {

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

      override def backward(delta: Delta): Unit = {
        synchronized {
          val (i, (j, value)) = delta
          // Cannot use += because of https://issues.scala-lang.org/browse/SI-10021
          upstreamDelta.value(i, j) = upstreamDelta.value(i, j) + value.value
        }
      }

      override val value: Data = {
        val ndarray = upstream.value.value
        val doubleArray = ndarray.data.asDouble()
        for (i <- (0 until ndarray.rows).view) yield {
          doubleArray.view(i * ndarray.columns, (i + 1) * ndarray.columns).map { doubleValue =>
            Eval.now(doubleValue)
          }
        }
      }
    }
  }
}
