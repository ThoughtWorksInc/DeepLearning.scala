package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Tape.Literal
import com.thoughtworks.raii.future.Do
import com.thoughtworks.raii.ownership.implicits._
import com.thoughtworks.raii.ownership._
import shapeless.PolyDefns.{Case1, Case2}
import shapeless._

trait ToTapeTask[From] extends DepFn1[From] {
  type Data
  type Delta
  override final type Out = Do.Covariant[Borrowing[Tape.Aux[Data, Delta]]]
}

object ToTapeTask {

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

  trait Poly1 extends shapeless.Poly1 {
    @inline
    implicit def tapeTaskCase[Operand0, Data0, Delta0](
        implicit toTapeTask0: ToTapeTask.Aux[Operand0, Data0, Delta0],
        tapeTaskCase: Lazy[Case[Do.Covariant[Borrowing[Tape.Aux[Data0, Delta0]]]]]
    ): Case.Aux[Operand0, tapeTaskCase.value.Result] = {
      at { (operand0: Operand0) =>
        tapeTaskCase.value(toTapeTask0(operand0))
      }
    }
  }

  trait Poly2 extends shapeless.Poly2 {
    @inline
    implicit def tapeTaskCase[F, Operand0, Operand1, Data0, Delta0, Data1, Delta1](
        implicit toTapeTask0: ToTapeTask.Aux[Operand0, Data0, Delta0],
        toTapeTask1: ToTapeTask.Aux[Operand1, Data1, Delta1],
        tapeTaskCase: Lazy[
          Case[Do.Covariant[Borrowing[Tape.Aux[Data0, Delta0]]], Do.Covariant[Borrowing[Tape.Aux[Data1, Delta1]]]]]
    ): Case.Aux[Operand0, Operand1, tapeTaskCase.value.Result] = {
      at { (operand0: Operand0, operand1: Operand1) =>
        tapeTaskCase.value(toTapeTask0(operand0), toTapeTask1(operand1))
      }
    }
  }

}
