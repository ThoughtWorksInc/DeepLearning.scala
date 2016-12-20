package com.thoughtworks.deeplearning
package dsl

import com.thoughtworks.deeplearning.dsl.layers.Identity
import com.thoughtworks.deeplearning.Layer.Batch

import language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class BackPropagationType[Data0, Delta0] {
  type Data = Data0
  type Delta = Delta0

  private type ConcreteBatch = Batch.Aux[Data, Delta]

  // Workaround for https://issues.scala-lang.org/browse/SI-10008
  type Batch >: ConcreteBatch <: ConcreteBatch

  type To[OutputSymbol <: BackPropagationType[_, _]] = Layer.Aux[Batch, OutputSymbol#Batch]
  //  type Layer.Aux[OutputType <: BackPropagationType] =
  //    Layer.Aux[ConcreteBatch, outputType.ConcreteBatch forSome { val outputType: OutputType }]
}

object BackPropagationType {

  implicit def apply[Data, Delta]: BackPropagationType[Data, Delta] = new BackPropagationType

  type DataOf[T <: BackPropagationType[_, _]] = t.Data forSome { val t: T }
  type DeltaOf[T <: BackPropagationType[_, _]] = t.Delta forSome { val t: T }

  implicit def inputTypeToLayer[InputData, InputDelta]: ToLayer.Aux[BackPropagationType[InputData, InputDelta],
                                                                    Batch.Aux[InputData, InputDelta],
                                                                    InputData,
                                                                    InputDelta] =
    new ToLayer[BackPropagationType[InputData, InputDelta], Batch.Aux[InputData, InputDelta]] {
      override type OutputData = InputData
      override type OutputDelta = InputDelta

      override def apply(input: BackPropagationType[InputData, InputDelta]) =
        Identity[InputData, InputDelta]()
    }

}
