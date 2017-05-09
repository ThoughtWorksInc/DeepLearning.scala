package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Tape
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import com.thoughtworks.raii.asynchronous
import com.thoughtworks.raii.ownership.{Borrowing, garbageCollectable}

package object differentiable {
  type INDArray = Do[Borrowing[Tape.Aux[org.nd4j.linalg.api.ndarray.INDArray, org.nd4j.linalg.api.ndarray.INDArray]]]
  type Float = Do[Borrowing[Tape.Aux[scala.Float, scala.Float]]]
  type Double = Do[Borrowing[Tape.Aux[scala.Double, scala.Double]]]
  type Any = Do[_ <: Tape.Aux[scala.Any, scala.Nothing]]
}
