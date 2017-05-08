package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Tape.Literal
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.ownership._
import com.thoughtworks.raii.ownership._
import shapeless.PolyDefns.{Case1, Case2}
import shapeless._

trait ToTapeTask[From] extends DepFn1[From] {
  type Data
  type Delta
  override final type Out = Do.Covariant[Borrowing[Tape.Aux[Data, Delta]]]
}

private[deeplearning] trait LowPriorityToTapeTaskFunctions { this: ToTapeTask.type =>

  @inline
  implicit def fromData[From, Delta0]: LowPriorityToTapeTask.Aux[From, From, Delta0] = {
    new LowPriorityToTapeTask[From] {
      override type Data = From
      override type Delta = Delta0

      @inline
      override def apply(a: Data): Out = {
        val myLiteral = garbageCollectable(Literal(a))
        Do.now(myLiteral)
      }
    }
  }
}

object ToTapeTask extends LowPriorityToTapeTaskFunctions {

  trait LiftCase1[P <: Poly, From] extends DepFn1[From]
  object LiftCase1 {
    type Aux[P <: Poly, From, Out0] = LiftCase1[P, From] {
      type Out = Out0
    }
    implicit def liftCase1[P <: Poly, From, Data0, Delta0, Out0](
        implicit lift: ToTapeTask.Aux[From, Data0, Delta0],
        case1: shapeless.PolyDefns.Case1.Aux[P, Do.Covariant[Borrowing[Tape.Aux[Data0, Delta0]]], Out0])
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
        implicit lift0: ToTapeTask.Aux[Operand0, Data0, Delta0],
        lift1: ToTapeTask.Aux[Operand1, Data1, Delta1],
        case2: shapeless.PolyDefns.Case2.Aux[P,
                                             Do.Covariant[Borrowing[Tape.Aux[Data0, Delta0]]],
                                             Do.Covariant[Borrowing[Tape.Aux[Data1, Delta1]]],
                                             Out0]): LiftCase2.Aux[P, Operand0, Operand1, Out0] = {
      new LiftCase2[P, Operand0, Operand1] {
        override type Out = Out0

        override def apply(operand0: Operand0, operand1: Operand1): Out = case2(lift0(operand0), lift1(operand1))
      }
    }
  }

  trait LowPriorityToTapeTask[From] extends ToTapeTask[From]
  object LowPriorityToTapeTask {
    type Aux[From, Data0, Delta0] = LowPriorityToTapeTask[From] {
      type Data = Data0
      type Delta = Delta0
    }
  }

  type Aux[From, Data0, Delta0] = ToTapeTask[From] {
    type Data = Data0
    type Delta = Delta0
  }

  @inline
  implicit def fromSubtype[Data0, Delta0, From](
      implicit constraint: From <:< Do[_ <: Borrowing[Tape.Aux[Data0, Delta0]]])
    : LowPriorityToTapeTask.Aux[From, Data0, Delta0] = {
    new LowPriorityToTapeTask[From] {
      override type Data = Data0
      override type Delta = Delta0

      @inline
      override def apply(a: From): Out = a
    }
  }

  def apply[From](implicit toTapeTask: ToTapeTask[From]): ToTapeTask.Aux[From, toTapeTask.Data, toTapeTask.Delta] =
    toTapeTask

}
