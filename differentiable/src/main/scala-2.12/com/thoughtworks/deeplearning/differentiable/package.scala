package com.thoughtworks.deeplearning

import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.ownership.Borrowing

package object differentiable{
  type Float = Do[Borrowing[Tape.Aux[scala.Float, scala.Float]]]
  type Double = Do[Borrowing[Tape.Aux[scala.Double, scala.Double]]]
  type Any = Do[_ <: Borrowing[Tape.Aux[scala.Any, scala.Nothing]]]
}