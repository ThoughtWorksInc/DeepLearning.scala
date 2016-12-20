package com.thoughtworks.deeplearning.dsl

import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.Layer
import shapeless.DepFn1

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait ToLiteral[From] extends DepFn1[From] {

  type OutputData
  type OutputDelta

  type Out = Batch.Aux[OutputData, OutputDelta] with Layer.Aux[Batch, Batch.Aux[OutputData, OutputDelta]]

}

object ToLiteral {
  type Aux[From, OutputData0, OutputDelta0] = ToLiteral[From] {
    type OutputData = OutputData0
    type OutputDelta = OutputDelta0
  }
}
