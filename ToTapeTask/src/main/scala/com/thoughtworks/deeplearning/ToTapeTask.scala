package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Tape.Literal
import com.thoughtworks.raii.future.Do
import shapeless.PolyDefns.{Case1, Case2}
import shapeless._

trait ToTapeTask[From] extends DepFn1[From] {
  type Data
  type Delta
  override final type Out = Do.Covariant[Tape.Aux[Data, Delta]]
}

object ToTapeTask {

  type Aux[From, Data0, Delta0] = ToTapeTask[From] {
    type Data = Data0
    type Delta = Delta0
  }

  @inline
  implicit def fromTape[Data0, Delta0, From <: Tape.Aux[Data0, Delta0]]: ToTapeTask.Aux[From, Data0, Delta0] = {
    new ToTapeTask[From] {
      override type Data = Data0
      override type Delta = Delta0

      @inline
      override def apply(a: From): Out = Do.now(a)
    }

  }

  @inline
  implicit def fromSubtype[Data0, Delta0, From <: Do[_ <: Tape.Aux[Data0, Delta0]]]
    : ToTapeTask.Aux[From, Data0, Delta0] = {
    new ToTapeTask[From] {
      override type Data = Data0
      override type Delta = Delta0

      @inline
      override def apply(a: From): Out = a
    }
  }

  @inline
  implicit def fromData[From, Delta0]: ToTapeTask.Aux[From, From, Delta0] = {
    new ToTapeTask[From] {
      override type Data = From
      override type Delta = Delta0

      @inline
      override def apply(a: Data): Out = Do.now(Literal(a))
    }

  }

  trait Poly1 extends shapeless.Poly1 {
    @inline
    implicit def tapeTaskCase[Operand0, Data0, Delta0](
        implicit toTapeTask0: ToTapeTask.Aux[Operand0, Data0, Delta0],
        tapeTaskCase: Lazy[Case[Do.Covariant[Tape.Aux[Data0, Delta0]]]]
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
        tapeTaskCase: Lazy[Case[Do.Covariant[Tape.Aux[Data0, Delta0]], Do.Covariant[Tape.Aux[Data1, Delta1]]]]
    ): Case.Aux[Operand0, Operand1, tapeTaskCase.value.Result] = {
      at { (operand0: Operand0, operand1: Operand1) =>
        tapeTaskCase.value(toTapeTask0(operand0), toTapeTask1(operand1))
      }
    }
  }

}
