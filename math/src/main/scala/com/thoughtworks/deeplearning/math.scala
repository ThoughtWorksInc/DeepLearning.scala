package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.ToTapeTask.{LiftCase1, LiftCase2}
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.ownership.Borrowing
import shapeless.{DepFn1, DepFn2, Poly, Poly1, Poly2}

/**
  * A namespace of definitions of polymophic functions.
  *
  * Those functions are implemented in other objects.
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object math {

  implicit final class PolyOps[Operand0](operand0: Operand0) {
    def -[Operand1](operand1: Operand1)(
        implicit methodCase: LiftCase2[polyFunctions.-.type, Operand0, Operand1]): methodCase.Out =
      methodCase(operand0, operand1)

    def +[Operand1](operand1: Operand1)(
        implicit methodCase: LiftCase2[polyFunctions.+.type, Operand0, Operand1]): methodCase.Out =
      methodCase(operand0, operand1)

    def *[Operand1](operand1: Operand1)(
        implicit methodCase: LiftCase2[polyFunctions.*.type, Operand0, Operand1]): methodCase.Out =
      methodCase(operand0, operand1)

    def /[Operand1](operand1: Operand1)(
        implicit methodCase: LiftCase2[polyFunctions./.type, Operand0, Operand1]): methodCase.Out =
      methodCase(operand0, operand1)
  }

  object polyFunctions {
    object - extends Poly2
    object + extends Poly2
    object * extends Poly2
    object / extends Poly2
    object abs extends Poly1
    object exp extends Poly1
    object log extends Poly1
    object max extends Poly2
    object min extends Poly2
  }

  def abs[From](a: From)(implicit liftCase1: LiftCase1[polyFunctions.abs.type, From]): liftCase1.Out = liftCase1(a)

  def exp[From](a: From)(implicit liftCase1: LiftCase1[polyFunctions.exp.type, From]): liftCase1.Out = liftCase1(a)

  def log[From](a: From)(implicit liftCase1: LiftCase1[polyFunctions.log.type, From]): liftCase1.Out = liftCase1(a)

  def max[Operand0, Operand1](operand0: Operand0, operand1: Operand1)(
      implicit methodCase: LiftCase2[polyFunctions.max.type, Operand0, Operand1]): methodCase.Out =
    methodCase(operand0, operand1)

  def min[Operand0, Operand1](operand0: Operand0, operand1: Operand1)(
      implicit methodCase: LiftCase2[polyFunctions.min.type, Operand0, Operand1]): methodCase.Out =
    methodCase(operand0, operand1)
}
