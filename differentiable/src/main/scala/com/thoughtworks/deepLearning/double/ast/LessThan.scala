package com.thoughtworks.deepLearning
package double.ast

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.Ast.{Cached, WidenAst}
import com.thoughtworks.deepLearning.Batch.WidenBatch
import com.thoughtworks.deepLearning.boolean.utilities.BooleanMonoidBatch
import com.thoughtworks.deepLearning.double.utilities.DoubleMonoidBatch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class LessThan[Input0 <: Batch](
    leftOperand: WidenAst[Input0, WidenBatch[Eval[scala.Double], Eval[scala.Double]]],
    rightOperand: WidenAst[Input0, WidenBatch[Eval[scala.Double], Eval[scala.Double]]]
) extends Cached {

  protected final class SharedBatch private[deepLearning](override val input: Input0,
                                    upstream1: WidenBatch[Eval[scala.Double], Eval[scala.Double]],
                                    upstream2: WidenBatch[Eval[scala.Double], Eval[scala.Double]])
      extends MonoidBatch
      with BooleanMonoidBatch {
    type Input >: Input0
    val value = upstream1.value.map2(upstream2.value)(_ < _).memoize

    override protected def closeUpstreams(): Unit = {
      upstream1.close()
      upstream2.close()
    }

    override protected def rawBackward(delta: Eval[scala.Boolean]): Unit = {
      upstream1.backward(Eval.now(0.0))
      upstream2.backward(Eval.now(0.0))
    }
  }

  type Input = Input0

  override protected def rawForward(input: Input): SharedBatch = {
    new SharedBatch(input, leftOperand.forward(input), rightOperand.forward(input))
  }
}
