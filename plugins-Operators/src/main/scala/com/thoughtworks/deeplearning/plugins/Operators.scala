package com.thoughtworks.deeplearning
package plugins
import shapeless.{Poly1, Poly2}

/** A namespace of definitions of polymorphic operators.
  *
  * Those functions are implemented in [[RawFloatLayers]], [[RawDoubleLayers]] and [[RawINDArrayLayers]].
  * @author 杨博 (Yang Bo)
  */
trait Operators {

  def abs[Operand0](operand0: Operand0)(implicit functionCase: Operators.abs.Case[Operand0]) = {
    functionCase(operand0)
  }
  def exp[Operand0](operand0: Operand0)(implicit functionCase: Operators.exp.Case[Operand0]) = {
    functionCase(operand0)
  }
  def log[Operand0](operand0: Operand0)(implicit functionCase: Operators.log.Case[Operand0]) = {
    functionCase(operand0)
  }
  def min[Operand0, Operand1](operand0: Operand0, operand1: Operand1)(
      implicit functionCase: Operators.min.Case[Operand0, Operand1]) = {
    functionCase(operand0, operand1)
  }
  def max[Operand0, Operand1](operand0: Operand0, operand1: Operand1)(
      implicit functionCase: Operators.max.Case[Operand0, Operand1]) = {
    functionCase(operand0, operand1)
  }

  trait ImplicitsApi {

    implicit final class PolymorphicOps[Operand0](operand0: Operand0) {
      def +[Operand1](operand1: Operand1)(
          implicit methodCase: Operators.+.Case[Operand0, Operand1]): methodCase.Result =
        methodCase(operand0, operand1)
      def -[Operand1](operand1: Operand1)(
          implicit methodCase: Operators.-.Case[Operand0, Operand1]): methodCase.Result =
        methodCase(operand0, operand1)
      def *[Operand1](operand1: Operand1)(
          implicit methodCase: Operators.*.Case[Operand0, Operand1]): methodCase.Result =
        methodCase(operand0, operand1)
      def /[Operand1](operand1: Operand1)(
          implicit methodCase: Operators./.Case[Operand0, Operand1]): methodCase.Result =
        methodCase(operand0, operand1)

    }
  }
  type Implicits <: ImplicitsApi

}

object Operators {

  object abs extends Poly1
  object exp extends Poly1
  object log extends Poly1
  object max extends Poly2
  object min extends Poly2

  object + extends Poly2
  object - extends Poly2
  object * extends Poly2
  object / extends Poly2

}
