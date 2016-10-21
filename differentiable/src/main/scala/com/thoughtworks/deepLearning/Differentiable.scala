package com.thoughtworks.deepLearning

import cats.{Eval, Monoid}
import cats.implicits._
import com.thoughtworks.deepLearning.Batch.Aux
import com.thoughtworks.deepLearning.Differentiable.Aux
import org.nd4j.linalg.api.ndarray.INDArray

trait Optimizer {

  type Data
  type Delta

  def applyPatch(data: Data, delta: Delta): Data

}

object Optimizer {

  type Aux[Data0, Delta0] = Optimizer {
    type Data = Data0
    type Delta = Delta0
  }

}

trait Differentiable extends Product {

  type Data
  type Delta

  type LiftFrom

  def lift(from: LiftFrom): Data

  def monoid: Monoid[Delta]

  final def id = DifferentiableFunction.Id[Data, Delta]()

  final def literal(initialValue: LiftFrom) = {
    DifferentiableFunction.Literal(lift(initialValue), this: Differentiable.Aux[Data, Delta])
  }

  final def weight(initialValue: LiftFrom)(implicit optimizer: Optimizer.Aux[Data, Delta]) = {
    DifferentiableFunction.Weight(lift(initialValue), this: Differentiable.Aux[Data, Delta], optimizer)
  }

}

object Differentiable {

  type Aux[Data0, Delta0] = Differentiable {
    type Data = Data0
    type Delta = Delta0
  }

  case object Double extends Differentiable {
    override type Data = Eval[scala.Double]
    override type Delta = Eval[scala.Double]
    override type LiftFrom = scala.Double
    override def lift(from: scala.Double) = Eval.now(from)
    override def monoid: Monoid[Delta] = implicitly
  }

}


trait Batch {

  type Data
  type Delta

  def value: Data

  def backward(delta: Delta): Unit

}

object Batch {

  type Aux[+Data0, -Delta0] = Batch {
    type Data <: Data0
    type Delta >: Delta0
  }

  final case class Literal[Data0](value: Data0) extends Batch {
    type Data = Data0
    type Delta >: Any

    def backward(delta: Delta): Unit = {}
  }

}

trait DifferentiableFunction {

  type InputData
  type InputDelta
  type OutputData
  type OutputDelta

  def forward(input: Batch.Aux[InputData, InputDelta]): Batch.Aux[OutputData, OutputDelta]

  def differentiable(inputDifferentiable: Differentiable.Aux[InputData, InputDelta]): Differentiable.Aux[OutputData, OutputDelta]

}

object DifferentiableFunction {

  type Aux[-InputData0, +InputDelta0, +OutputData0, -OutputDelta0] = DifferentiableFunction {
    type InputData >: InputData0
    type InputDelta <: InputDelta0
    type OutputData <: OutputData0
    type OutputDelta >: OutputDelta0
  }


  final case class Id[Data, Delta]() extends DifferentiableFunction {

    override type InputData = Data
    override type OutputData = Data
    override type InputDelta = Delta
    override type OutputDelta = Delta

    def forward(input: Batch.Aux[Data, Delta]): Batch.Aux[Data, Delta] = input

    def differentiable(input: Differentiable.Aux[Data, Delta]): Differentiable.Aux[Data, Delta] = input

  }

  final case class Literal[Data0, Delta0](override val value: Data0, differentiable: Differentiable.Aux[Data0, Delta0]) extends DifferentiableFunction with Batch {
    override type InputData = Any
    override type InputDelta = Nothing
    override type OutputData = Data0
    override type OutputDelta = Delta0
    override type Data = Data0
    override type Delta = Delta0

    override def backward(delta: Delta): Unit = {}

    override def forward(input: Batch.Aux[InputData, InputDelta]) = this

    override def differentiable(inputDifferentiable: Differentiable.Aux[InputData, InputDelta]) = differentiable

  }

  // TODO: thread safety
  final case class Weight[Data0, Delta0](var value: Data0, differentiable: Differentiable.Aux[Data0, Delta0], optimizer: Optimizer.Aux[Data0,Delta0]) extends DifferentiableFunction with Batch {
    override type InputData = Any
    override type InputDelta = Nothing
    override type OutputData = Data0
    override type OutputDelta = Delta0
    override type Data = Data0
    override type Delta = Delta0

    override def backward(delta: Delta): Unit = {
      value = optimizer.applyPatch(value, delta)
    }

    override def forward(input: Batch.Aux[InputData, InputDelta]) = this

    override def differentiable(inputDifferentiable: Differentiable.Aux[InputData, InputDelta]) = differentiable

  }

  final case class Add[InputData0, InputDelta0](
    left: DifferentiableFunction.Aux[InputData0, InputDelta0, Eval[scala.Double], Eval[scala.Double]],
    right: DifferentiableFunction.Aux[InputData0, InputDelta0, Eval[scala.Double], Eval[scala.Double]]
  ) extends DifferentiableFunction {
    override type InputData = InputData0
    override type InputDelta = InputDelta0
    override type OutputData = Eval[scala.Double]
    override type OutputDelta = Eval[scala.Double]

    override def forward(input: Batch.Aux[InputData, InputDelta]) = {
      val leftBatch = left.forward(input)
      val rightBatch = right.forward(input)
      // TODO: cache
      new Batch {
        type Data = Eval[scala.Double]
        type Delta = Eval[scala.Double]
        val value = leftBatch.value.map2(rightBatch.value) { _ + _ }.memoize
        override def backward(delta: Delta): Unit = {
          leftBatch.backward(delta)
          rightBatch.backward(delta)
        }
      }
    }

    override def differentiable(inputDifferentiable: Differentiable.Aux[InputData, InputDelta]) = Differentiable.Double

  }

}

// final case class DoubleLiteral(val value: ) extends DifferentiableFunction with Batch {
//   override type InputData = Any
//   override type InputDelta = Nothing
//   override type Data = Data0
//   override type Delta = Delta0
//   override type OutputData = Data
//   override type OutputDelta = Delta
//
//   def forward(input: Batch.Aux[InputData, InputDelta]): Batch.Aux[OutputData, OutputDelta] = {
//     this
//   }
//
//   def differentiable(inputDifferentiable: Differentiable.Aux[InputData, InputDelta]): Differentiable.Aux[OutputData, OutputDelta]
// }
