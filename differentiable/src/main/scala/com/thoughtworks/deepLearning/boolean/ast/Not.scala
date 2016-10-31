package com.thoughtworks.deepLearning.boolean.ast

import cats._
import cats.implicits._
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning.{Batch, Differentiable}
import com.thoughtworks.deepLearning.Differentiable._
import com.thoughtworks.deepLearning.any.utilities.Cached
import com.thoughtworks.deepLearning.boolean.utilities.BooleanMonoidBatch
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Not[Input0 <: Batch](
                                       differentiableBoolean: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]])
  extends Cached {

  protected final class SharedBatch(override val input: Input0,
                                    upstream: Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]])
    extends MonoidBatch
      with BooleanMonoidBatch {
    type Input >: Input0
    val value = upstream.value.map(!_)

    override protected def rawBackward(delta: Eval[scala.Boolean]): Unit = {
      upstream.backward(delta.map(!_))
    }

    override protected def closeUpstreams(): Unit = {
      upstream.close()
    }
  }

  type Input = Input0

  override protected def rawForward(input: Input): SharedBatch = {
    val upstream = differentiableBoolean.forward(input)
    new SharedBatch(input, upstream)
  }
}
