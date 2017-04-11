package com.thoughtworks.deeplearning

import com.thoughtworks.raii.{RAIITask, RAIITask2}
import shapeless.{Lazy, Poly1, Poly2}

/**
  * A namespace of common math operators.
  *
  * [[Poly.MathMethods MathMethods]] and [[Poly.MathFunctions MathFunctions]] provide functions like [[Poly.MathMethods.+ +]], [[Poly.MathMethods.- -]], [[Poly.MathMethods.* *]], [[Poly.MathMethods./ /]],
  * [[Poly.MathFunctions.log log]], [[Poly.MathFunctions.abs abs]], [[Poly.MathFunctions.max max]], [[Poly.MathFunctions.min min]] and [[Poly.MathFunctions.exp exp]], those functions been implements in specific Differentiable Object such as [[???]]
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object Poly {

  object MathMethods {
    object - extends Poly2
    object + extends Poly2 {
      implicit def raiiTaskCase[Operand0, Operand1, Tape0, Tape1](
          implicit toRaiiTask0: Operand0 => RAIITask2[Tape0],
          toRaiiTask1: Operand1 => RAIITask2[ Tape1],
          raiiTaskCase: Lazy[Case[RAIITask2[Tape0], RAIITask2[Tape1]]]
      ): Case.Aux[Operand0, Operand1, raiiTaskCase.value.Result] = {
        at { (operand0: Operand0, operand1: Operand1) =>
          type Task0 = RAIITask2[Tape0]
          type Task1 = RAIITask2[Tape1]
          val task0: Task0 = toRaiiTask0(operand0)
          val task1: Task1 = toRaiiTask1(operand1)

          def forceApply[A, B](lazyCase: Lazy[Case[A, B]], a: A, b: B): lazyCase.value.Result = {
            lazyCase.value(a, b)
          }
          forceApply[Task0, Task1](raiiTaskCase, task0, task1)
        }
      }
    }
    object * extends Poly2
    object / extends Poly2
  }

  implicit final class MathOps[Left](left: Left) {
    def -[Right](right: Right)(implicit methodCase: MathMethods.-.Case[Left, Right]): methodCase.Result =
      MathMethods.-(left, right)

    def +[Right](right: Right)(implicit methodCase: MathMethods.+.Case[Left, Right]): methodCase.Result =
      MathMethods.+(left, right)

    def *[Right](right: Right)(implicit methodCase: MathMethods.*.Case[Left, Right]): methodCase.Result =
      MathMethods.*(left, right)

    def /[Right](right: Right)(implicit methodCase: MathMethods./.Case[Left, Right]): methodCase.Result =
      MathMethods./(left, right)
  }

  object MathFunctions {

    object log extends Poly1
    object exp extends Poly1
    object abs extends Poly1
    object max extends Poly2
    object min extends Poly2

  }

}
