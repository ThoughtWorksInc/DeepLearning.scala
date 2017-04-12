package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Tape.Literal
import com.thoughtworks.raii.RAIITask
import shapeless.PolyDefns.{Case1, Case2}
import shapeless._

/**
  * A namespace of common math operators for [[RAIITask]] of [[Tape]]s.
  *
  * [[Poly.MathMethods MathMethods]] and [[Poly.MathFunctions MathFunctions]] provide functions like [[Poly.MathMethods.+ +]], [[Poly.MathMethods.- -]], [[Poly.MathMethods.* *]], [[Poly.MathMethods./ /]],
  * [[Poly.MathFunctions.log log]], [[Poly.MathFunctions.abs abs]], [[Poly.MathFunctions.max max]], [[Poly.MathFunctions.min min]] and [[Poly.MathFunctions.exp exp]], those functions been implements in specific Differentiable Object such as [[???]]
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object Poly {

  trait ToTapeTask[From] extends DepFn1[From] {
    type Data
    type Delta
    override final type Out = RAIITask.Covariant[Tape.Aux[Data, Delta]]
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
        override def apply(a: From): Out = RAIITask.unmanaged(a)
      }

    }

    @inline
    implicit def fromSubtype[Data0, Delta0, From <: RAIITask[_ <: Tape.Aux[Data0, Delta0]]]
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
        override def apply(a: Data): Out = RAIITask.unmanaged(Literal(a))
      }

    }

    trait Poly1 extends shapeless.Poly1 {
      @inline
      implicit def tapeTaskCase[Operand0, Data0, Delta0](
          implicit toTapeTask0: ToTapeTask.Aux[Operand0, Data0, Delta0],
          tapeTaskCase: Lazy[Case[RAIITask.Covariant[Tape.Aux[Data0, Delta0]]]]
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
            Case[RAIITask.Covariant[Tape.Aux[Data0, Delta0]], RAIITask.Covariant[Tape.Aux[Data1, Delta1]]]]
      ): Case.Aux[Operand0, Operand1, tapeTaskCase.value.Result] = {
        at { (operand0: Operand0, operand1: Operand1) =>
          tapeTaskCase.value(toTapeTask0(operand0), toTapeTask1(operand1))
        }
      }
    }

  }

  object MathMethods {
    object - extends ToTapeTask.Poly2
    object + extends ToTapeTask.Poly2
    object * extends ToTapeTask.Poly2
    object / extends ToTapeTask.Poly2
  }

  implicit final class MathOps[Operand0](operand0: Operand0) {
    @inline
    def -[Operand1](operand1: Operand1)(
        implicit methodCase: MathMethods.-.Case[Operand0, Operand1]): methodCase.Result =
      methodCase(operand0, operand1)

    @inline
    def +[Operand1](operand1: Operand1)(
        implicit methodCase: MathMethods.+.Case[Operand0, Operand1]): methodCase.Result =
      methodCase(operand0, operand1)

    @inline
    def *[Operand1](operand1: Operand1)(
        implicit methodCase: MathMethods.*.Case[Operand0, Operand1]): methodCase.Result =
      methodCase(operand0, operand1)

    @inline
    def /[Operand1](operand1: Operand1)(
        implicit methodCase: MathMethods./.Case[Operand0, Operand1]): methodCase.Result =
      methodCase(operand0, operand1)
  }

  object MathFunctions {

    object log extends ToTapeTask.Poly1
    object exp extends ToTapeTask.Poly1
    object abs extends ToTapeTask.Poly1
    object max extends ToTapeTask.Poly2
    object min extends ToTapeTask.Poly2

  }

}
