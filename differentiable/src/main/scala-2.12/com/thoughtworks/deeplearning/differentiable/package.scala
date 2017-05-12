package com.thoughtworks.deeplearning

import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.ownership.Borrowing

package object differentiable{
  type Float = Do[Borrowing[Tape[scala.Float, scala.Float]]]
  type Double = Do[Borrowing[Tape[scala.Double, scala.Double]]]
  type Any = Do[ Borrowing[Tape[scala.Any, scala.Nothing]]]
}