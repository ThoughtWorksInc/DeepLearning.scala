package com.thoughtworks.deepLearning.boolean.ast

import cats._
import cats.implicits._
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning.{Batch, Differentiable}
import com.thoughtworks.deepLearning.Differentiable._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class If[Input0 <: Batch, Output0 <: Batch](
                                                        condition: Differentiable.Aux[Input0, Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]],
                                                        `then`: Differentiable.Aux[Input0, Output0],
                                                        `else`: Differentiable.Aux[Input0, Output0])
  extends Differentiable {
  override type Input = Input0
  override type Output = Output0

  override def forward(input: Input0): Output0 = {
    val conditionForwardPass = condition.forward(input)
    if (conditionForwardPass.value.value) {
      `then`.forward(input)
    } else {
      `else`.forward(input)
    }
  }
}