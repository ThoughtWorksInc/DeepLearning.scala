package com.thoughtworks.deeplearning

/**
  * A namespace of definitions of polymophic functions.
  *
  * Those functions are implemented in other objects.
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object PolyFunctions {

  object PolyMethods {
    object - extends ToTapeTask.Poly2
    object + extends ToTapeTask.Poly2
    object * extends ToTapeTask.Poly2
    object / extends ToTapeTask.Poly2
  }

  implicit final class PolyOps[Operand0](operand0: Operand0) {
    def -[Operand1](operand1: Operand1)(
        implicit methodCase: PolyMethods.-.Case[Operand0, Operand1]): methodCase.Result =
      methodCase(operand0, operand1)

    def +[Operand1](operand1: Operand1)(
        implicit methodCase: PolyMethods.+.Case[Operand0, Operand1]): methodCase.Result =
      methodCase(operand0, operand1)

    def *[Operand1](operand1: Operand1)(
        implicit methodCase: PolyMethods.*.Case[Operand0, Operand1]): methodCase.Result =
      methodCase(operand0, operand1)

    def /[Operand1](operand1: Operand1)(
        implicit methodCase: PolyMethods./.Case[Operand0, Operand1]): methodCase.Result =
      methodCase(operand0, operand1)
  }

  object log extends ToTapeTask.Poly1
  object exp extends ToTapeTask.Poly1
  object abs extends ToTapeTask.Poly1
  object max extends ToTapeTask.Poly2
  object min extends ToTapeTask.Poly2

}
