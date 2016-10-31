package com.thoughtworks.deepLearning
package double.ast

import cats._
import cats.implicits._
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning.Differentiable
import com.thoughtworks.deepLearning.Differentiable._
import com.thoughtworks.deepLearning.any.utilities.Cached
import com.thoughtworks.deepLearning.double.utilities.DoubleMonoidBatch
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Reciprocal[Input0 <: Batch](
    operand: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]])
    extends Cached {

  protected final class SharedBatch(override val input: Input0,
                                    upstream: Batch.Aux[Eval[scala.Double], Eval[scala.Double]])
      extends MonoidBatch
      with DoubleMonoidBatch {
    type Input >: Input0
    val value = upstream.value.map(1.0 / _)

    override protected def closeUpstreams(): Unit = {
      upstream.close()
    }

    override protected def rawBackward(delta: Eval[scala.Double]): Unit = {
      val a = upstream.value
      upstream.backward(delta.map2(a) { (outputDeltaValue: scala.Double, aValue: scala.Double) =>
        -outputDeltaValue / (aValue * aValue)
      })
    }
  }

  type Input = Input0

  override protected def rawForward(input: Input): SharedBatch = {
    val upstream = operand.forward(input)
    new SharedBatch(input, upstream)
  }
}
