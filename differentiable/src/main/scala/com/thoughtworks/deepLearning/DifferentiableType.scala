package com.thoughtworks.deepLearning

import scala.language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
sealed trait DifferentiableType {
  type Data
  type Delta

  private[deepLearning] type ConcreteBatch = Differentiable.Batch[Data, Delta]

  // Workaround for https://issues.scala-lang.org/browse/SI-10008
  type Batch >: ConcreteBatch <: ConcreteBatch

  type Ast[OutputSymbol <: DifferentiableType] = DifferentiableFunction.Ast[ConcreteBatch, OutputSymbol#Batch]
//  type Ast[OutputType <: DifferentiableType] =
//    DifferentiableFunction.Ast[ConcreteBatch, outputType.ConcreteBatch forSome { val outputType: OutputType }]
}

object DifferentiableType {

  type Aux[Data0, Delta0] = DifferentiableType {
    type Data = Data0
    type Delta = Delta0

  }

  final class ConcreteType[Data0, Delta0] extends DifferentiableType { this: Aux[Data0, Delta0] =>
    type Data = Data0
    type Delta = Delta0
  }

  implicit def apply[Data, Delta]: ConcreteType[Data, Delta] = new ConcreteType

  type OfBatch[Batch0 <: Differentiable] = DifferentiableType {
    type ConcreteBatch = Batch0
  }
}
