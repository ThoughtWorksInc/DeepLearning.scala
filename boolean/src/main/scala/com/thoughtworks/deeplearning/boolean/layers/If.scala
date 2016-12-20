package com.thoughtworks.deeplearning.boolean.layers

import com.thoughtworks.deeplearning.{Batch, Layer}
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.boolean.utilities._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class If[Input0 <: Batch, OutputData0, OutputDelta0](
                                                                 condition: Layer.Aux[Input0, BpBoolean#Batch],
                                                                 `then`: Layer.Aux[Input0, Batch.Aux[OutputData0, OutputDelta0]],
                                                                 `else`: Layer.Aux[Input0, Batch.Aux[OutputData0, OutputDelta0]])
    extends Layer {
  override type Input = Input0
  override type Output = Batch.Aux[OutputData0, OutputDelta0]

  override def forward(input: Input0) = {
    resource.managed(condition.forward(input)).acquireAndGet { conditionBatch =>
      (if (conditionBatch.value.value) `then` else `else`).forward(input)
    }
  }
}
