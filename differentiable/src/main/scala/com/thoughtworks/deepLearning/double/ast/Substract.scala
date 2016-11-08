package com.thoughtworks.deepLearning
package double.ast

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.DifferentiableFunction._
import com.thoughtworks.deepLearning.double.utilities.DoubleMonoidBatch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Substract[Input0 <: Differentiable](
    leftOperand: DifferentiableFunction.Ast[Input0, Differentiable.Batch[Eval[scala.Double], Eval[scala.Double]]],
    rightOperand: DifferentiableFunction.Ast[Input0, Differentiable.Batch[Eval[scala.Double], Eval[scala.Double]]]
) extends Cached {

  protected final class SharedBatch private[deepLearning] (
      override val input: Input0,
      upstream1: Differentiable.Batch[Eval[scala.Double], Eval[scala.Double]],
      upstream2: Differentiable.Batch[Eval[scala.Double], Eval[scala.Double]])
      extends MonoidBatch
      with DoubleMonoidBatch {
    type Input >: Input0
    val value = upstream1.value.map2(upstream2.value)(_ - _)

    override protected def closeUpstreams(): Unit = {
      upstream1.close()
      upstream2.close()
    }

    override protected def rawBackward(delta: Eval[scala.Double]): Unit = {
      upstream1.backward(delta)
      upstream2.backward(delta.map(-_))
    }
  }

  type Input = Input0

  override protected def rawForward(input: Input): SharedBatch = {
    new SharedBatch(input, leftOperand.forward(input), rightOperand.forward(input))
  }

}
