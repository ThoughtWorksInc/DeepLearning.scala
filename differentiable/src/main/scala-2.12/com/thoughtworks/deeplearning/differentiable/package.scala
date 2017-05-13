package com.thoughtworks.deeplearning

import com.thoughtworks.raii.asynchronous.Do

package object differentiable{
  type Float = Do[Tape[scala.Float, scala.Float]]
  type Double = Do[Tape[scala.Double, scala.Double]]
  type Any = Do[ Tape[scala.Any, scala.Nothing]]
}