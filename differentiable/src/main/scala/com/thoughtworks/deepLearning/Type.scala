package com.thoughtworks.deepLearning

import scala.language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class Type[Data0, Delta0] {
  type Data = Data0
  type Delta = Delta0

  private[deepLearning] type ConcreteBatch = Batch.Aux[Data, Delta]

  // Workaround for https://issues.scala-lang.org/browse/SI-10008
  type Batch >: ConcreteBatch <: ConcreteBatch

  type To[OutputSymbol <: Type[_, _]] = NeuralNetwork.Aux[ConcreteBatch, OutputSymbol#Batch]
  //  type NeuralNetwork.Aux[OutputType <: Type] =
  //    NeuralNetwork.Aux[ConcreteBatch, outputType.ConcreteBatch forSome { val outputType: OutputType }]
}

object Type {

  implicit def apply[Data, Delta]: Type[Data, Delta] = new Type

}
