package com.thoughtworks.deepLearning
package double.ast

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.DifferentiableFunction.{Cached, Ast}
import com.thoughtworks.deepLearning.Differentiable.Batch
import com.thoughtworks.deepLearning.boolean.utilities.BooleanMonoidBatch
import com.thoughtworks.deepLearning.double.utilities.DoubleMonoidBatch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Multiply[Input0 <: Differentiable](
                                                     leftOperand: Ast[Input0, Batch[Eval[scala.Double], Eval[scala.Double]]],
                                                     rightOperand: Ast[Input0, Batch[Eval[scala.Double], Eval[scala.Double]]]
) extends Cached {

  protected final class SharedBatch private[deepLearning](override val input: Input0,
                                                          upstream1: Batch[Eval[scala.Double], Eval[scala.Double]],
                                                          upstream2: Batch[Eval[scala.Double], Eval[scala.Double]])
      extends MonoidBatch
      with DoubleMonoidBatch {
    type Input >: Input0
    val value = upstream1.value.map2(upstream2.value)(_ * _)

    override protected def closeUpstreams(): Unit = {
      upstream1.close()
      upstream2.close()
    }

    override protected def rawBackward(delta: Eval[scala.Double]): Unit = {
      val a = upstream1.value
      val b = upstream2.value
      upstream1.backward(delta.map2(b)(_ * _))
      upstream2.backward(delta.map2(a)(_ * _))
    }
  }

  type Input = Input0

  override protected def rawForward(input: Input): SharedBatch = {
    new SharedBatch(input, leftOperand.forward(input), rightOperand.forward(input))
  }

}
