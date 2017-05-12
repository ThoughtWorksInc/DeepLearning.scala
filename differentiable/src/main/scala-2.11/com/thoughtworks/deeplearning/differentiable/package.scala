package com.thoughtworks.deeplearning

import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.ownership.Borrowing

package object differentiable {
  type INDArray = Do[Borrowing[Tape[org.nd4j.linalg.api.ndarray.INDArray, org.nd4j.linalg.api.ndarray.INDArray]]]
  type Float = Do[Borrowing[Tape[scala.Float, scala.Float]]]
  type Double = Do[Borrowing[Tape[scala.Double, scala.Double]]]
  type Any = Do[ Borrowing[Tape[scala.Any, scala.Nothing]]]
}
