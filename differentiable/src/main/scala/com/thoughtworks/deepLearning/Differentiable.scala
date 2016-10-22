package com.thoughtworks.deepLearning

import cats.{Eval, Monoid, Semigroup}
import cats.implicits._
import com.thoughtworks.deepLearning.Batch.Aux
import com.thoughtworks.deepLearning.Metadata.Aux
import org.nd4j.linalg.api.ndarray.INDArray
import shapeless.{:+:, CNil, Coproduct, Inr}
import simulacrum.{noop, typeclass}

import scala.language.implicitConversions

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

@typeclass
trait Differentiable[-LiftFrom] extends Metadata {
  typeClassInstance =>

  def lift(from: LiftFrom): typeClassInstance.Data

  final def literal(initialValue: LiftFrom) = {
    DifferentiableFunction.Literal(lift(initialValue), this: Metadata.Aux[Data, Delta])
  }

  final def weight(initialValue: LiftFrom)(implicit optimizer: Optimizer.Aux[typeClassInstance.Data, typeClassInstance.Delta]) = {
    DifferentiableFunction.Weight(lift(initialValue), this: Metadata.Aux[Data, Delta], optimizer)
  }

  final def id = DifferentiableFunction.Id(this: Metadata.Aux[Data, Delta])

}

object Differentiable {

  implicit case object Double extends Differentiable[scala.Double] {
    override type Data = Eval[scala.Double]
    override type Delta = Eval[scala.Double]

    override def lift(from: scala.Double) = Eval.now(from)

    override val monoid: Monoid[Delta] = implicitly
  }

  implicit case object CNil extends Metadata.Coproduct with Differentiable[shapeless.CNil] {
    override type Data = shapeless.CNil
    override type DeltaValue = shapeless.CNil

    override object semigroup extends Semigroup[DeltaValue] {
      override def combine(x: CNil, y: CNil): CNil = x
    }

    override def lift(from: shapeless.CNil) = from
  }

  final case class CCons[LiftFromHead, LiftFromTail <: shapeless.Coproduct](head: Differentiable[LiftFromHead], tail: Differentiable[LiftFromTail] with Metadata.Coproduct) extends Metadata.Coproduct with Differentiable[shapeless.:+:[LiftFromHead, LiftFromTail]] {
    override type Data = shapeless.:+:[head.Data, tail.Data]
    override type DeltaValue = shapeless.:+:[head.Delta, tail.DeltaValue]

    override def lift(from: shapeless.:+:[LiftFromHead, LiftFromTail]) = {
      from match {
        case shapeless.Inl(fromHead) =>
          shapeless.Inl(head.lift(fromHead))
        case shapeless.Inr(fromTail) =>
          shapeless.Inr(tail.lift(fromTail))
      }
    }

    override def semigroup = new Semigroup[DeltaValue] {
      override def combine(x: DeltaValue, y: DeltaValue) = {
        (x, y) match {
          case (shapeless.Inr(xTail), shapeless.Inr(yTail)) =>
            shapeless.Inr(tail.semigroup.combine(xTail, yTail))
          case (shapeless.Inl(xHead), shapeless.Inl(yHead)) =>
            shapeless.Inl(head.monoid.combine(xHead, yHead))
          case _ =>
            throw new IllegalArgumentException("Deltas of a coproduct must have the same choice.")
        }
      }
    }
  }

  implicit def ccons[LiftFromHead, LiftFromTail <: shapeless.Coproduct](head: Differentiable[LiftFromHead], tail: Differentiable[LiftFromTail] with Metadata.Coproduct) = CCons[LiftFromHead, LiftFromTail](head, tail)

}

trait Metadata {

  type Data
  type Delta

  def monoid: Monoid[Delta]

}

object Metadata {

  type Aux[Data0, Delta0] = Metadata {
    type Data = Data0
    type Delta = Delta0
  }

  object Coproduct {
    type Aux[Data0, DeltaValue0] = Coproduct {
      type Data = Data0
      type DeltaValue = DeltaValue0
    }
  }

  sealed trait Coproduct extends Metadata {

    type Data <: shapeless.Coproduct
    type DeltaValue <: shapeless.Coproduct

    def semigroup: Semigroup[DeltaValue]

    def monoid = cats.instances.option.catsKernelStdMonoidForOption(semigroup)

    type Delta = Option[DeltaValue]
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

  def metadata: Metadata.Aux[OutputData, OutputDelta]

}

object DifferentiableFunction {

  type Aux[-InputData0, +InputDelta0, +OutputData0, -OutputDelta0] = DifferentiableFunction {
    type InputData >: InputData0
    type InputDelta <: InputDelta0
    type OutputData <: OutputData0
    type OutputDelta >: OutputDelta0
  }


  final case class Id[Data, Delta](override val metadata: Metadata.Aux[Data, Delta]) extends DifferentiableFunction {

    override type InputData = Data
    override type OutputData = Data
    override type InputDelta = Delta
    override type OutputDelta = Delta

    def forward(input: Batch.Aux[Data, Delta]): Batch.Aux[Data, Delta] = input

  }

  final case class Literal[Data0, Delta0](override val value: Data0, override val metadata: Metadata.Aux[Data0, Delta0]) extends DifferentiableFunction with Batch {
    override type InputData = Any
    override type InputDelta = Nothing
    override type OutputData = Data0
    override type OutputDelta = Delta0
    override type Data = Data0
    override type Delta = Delta0

    override def backward(delta: Delta): Unit = {}

    override def forward(input: Batch.Aux[InputData, InputDelta]) = this

  }

  // TODO: thread safety
  final case class Weight[Data0, Delta0](var value: Data0, override val metadata: Metadata.Aux[Data0, Delta0], optimizer: Optimizer.Aux[Data0, Delta0]) extends DifferentiableFunction with Batch {
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

  }

  final case class DoubleAddDouble[InputData0, InputDelta0]
  (
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
        type Data = OutputData
        type Delta = OutputDelta
        val value = leftBatch.value.map2(rightBatch.value) {
          _ + _
        }.memoize

        override def backward(delta: Delta): Unit = {
          leftBatch.backward(delta)
          rightBatch.backward(delta)
        }
      }
    }

    override def metadata = Differentiable.Double

  }

  final case class Inl[InputData0, InputDelta0, HeadData, HeadDelta, TailData <: shapeless.Coproduct, TailDeltaValue <: shapeless.Coproduct]
  (
    head: DifferentiableFunction.Aux[InputData0, InputDelta0, HeadData, HeadDelta],
    tailMetadata: Metadata.Coproduct
  ) extends DifferentiableFunction {
    override type InputData = InputData0
    override type InputDelta = InputDelta0

    override type OutputData = shapeless.Inl[head.OutputData, Nothing]
    override type OutputDelta = Option[shapeless.:+:[head.OutputData, shapeless.Coproduct]]

    override def forward(input: Batch.Aux[InputData, InputDelta]) = {
      val headBatch = head.forward(input)
      new Batch {
        type Data = OutputData
        type Delta = OutputDelta
        val value = ???

        override def backward(delta: Delta): Unit = ???
      }
      ???
    }

    // Prefer asInstanceOf instead of complicated type constraint
    override def metadata = {
      def asDifferentiable[LiftFrom](metadata: Metadata) = {
        metadata.asInstanceOf[Differentiable[LiftFrom]]
      }
      def asDifferentiableCoproduct[LiftFrom <: shapeless.Coproduct](metadata: Metadata) = {
        metadata.asInstanceOf[Differentiable[LiftFrom] with Metadata.Coproduct]
      }
      Differentiable.CCons(asDifferentiable(head.metadata), asDifferentiableCoproduct(tailMetadata)).asInstanceOf[Metadata.Aux[OutputData, OutputDelta]]
    }

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
//   def metadata: Metadata.Aux[OutputData, OutputDelta]
// }
