package com.thoughtworks.deeplearning

import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.ownership._
import com.thoughtworks.raii.ownership._
import shapeless.PolyDefns.{Case1, Case2}
import shapeless._

trait Lift[From] extends DepFn1[From] {
  type Data
  type Delta
  override final type Out = Do[Borrowing[Tape[Data, Delta]]]
}

private[deeplearning] trait LowPriorityLiftFunctions { this: Lift.type =>

  @inline
  implicit def fromData[From, Delta0]: LowPriorityLift.Aux[From, From, Delta0] = {
    new LowPriorityLift[From] {
      override type Data = From
      override type Delta = Delta0

      @inline
      override def apply(a: Data): Out = {
        val myLiteral = garbageCollectable(Tape.literal(a))
        Do.now(myLiteral)
      }
    }
  }
}

object Lift extends LowPriorityLiftFunctions {

  trait LiftCase1[P <: Poly, From] extends DepFn1[From]
  object LiftCase1 {
    type Aux[P <: Poly, From, Out0] = LiftCase1[P, From] {
      type Out = Out0
    }
    implicit def liftCase1[P <: Poly, From, Data0, Delta0, Out0](
        implicit lift: Lift.Aux[From, Data0, Delta0],
        case1: shapeless.PolyDefns.Case1.Aux[P, Do[Borrowing[Tape[Data0, Delta0]]], Out0])
      : LiftCase1.Aux[P, From, Out0] = {
      new LiftCase1[P, From] {
        override type Out = Out0

        override def apply(t: From): Out = case1(lift(t))
      }
    }
  }

  trait LiftCase2[P <: Poly, Operand0, Operand1] extends DepFn2[Operand0, Operand1]
  object LiftCase2 {
    type Aux[P <: Poly, Operand0, Operand1, Out0] = LiftCase2[P, Operand0, Operand1] {
      type Out = Out0
    }
    implicit def liftCase2[P <: Poly, Operand0, Operand1, Data0, Delta0, Data1, Delta1, Out0](
        implicit lift0: Lift.Aux[Operand0, Data0, Delta0],
        lift1: Lift.Aux[Operand1, Data1, Delta1],
        case2: shapeless.PolyDefns.Case2.Aux[P,
                                             Do[Borrowing[Tape[Data0, Delta0]]],
                                             Do[Borrowing[Tape[Data1, Delta1]]],
                                             Out0]): LiftCase2.Aux[P, Operand0, Operand1, Out0] = {
      new LiftCase2[P, Operand0, Operand1] {
        override type Out = Out0

        override def apply(operand0: Operand0, operand1: Operand1): Out = case2(lift0(operand0), lift1(operand1))
      }
    }
  }

  trait LowPriorityLift[From] extends Lift[From]
  object LowPriorityLift {
    type Aux[From, Data0, Delta0] = LowPriorityLift[From] {
      type Data = Data0
      type Delta = Delta0
    }
  }

  type Aux[From, Data0, Delta0] = Lift[From] {
    type Data = Data0
    type Delta = Delta0
  }

  @inline
  implicit def fromSubtype[Data0, Delta0, From](
      implicit constraint: From <:< Do[ Borrowing[Tape[Data0, Delta0]]])
    : LowPriorityLift.Aux[From, Data0, Delta0] = {
    new LowPriorityLift[From] {
      override type Data = Data0
      override type Delta = Delta0

      @inline
      override def apply(a: From): Out = a
    }
  }

  def apply[From](implicit lift: Lift[From]): Lift.Aux[From, lift.Data, lift.Delta] = lift

}
