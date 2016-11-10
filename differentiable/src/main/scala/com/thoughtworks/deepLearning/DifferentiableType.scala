package com.thoughtworks.deepLearning

import scala.language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DifferentiableType[Data0, Delta0] {
  type Data = Data0
  type Delta = Delta0

  private[deepLearning] type ConcreteBatch = Differentiable.Batch[Data, Delta]

  // Workaround for https://issues.scala-lang.org/browse/SI-10008
  type Batch >: ConcreteBatch <: ConcreteBatch

  type Ast[OutputSymbol <: DifferentiableType[_, _]] = DifferentiableFunction.Ast[ConcreteBatch, OutputSymbol#Batch]
  //  type Ast[OutputType <: DifferentiableType] =
  //    DifferentiableFunction.Ast[ConcreteBatch, outputType.ConcreteBatch forSome { val outputType: OutputType }]
}

object DifferentiableType {

  implicit def apply[Data, Delta]: DifferentiableType[Data, Delta] = new DifferentiableType

}
