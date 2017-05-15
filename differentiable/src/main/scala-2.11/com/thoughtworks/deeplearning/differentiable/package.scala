package com.thoughtworks.deeplearning

import com.thoughtworks.raii.asynchronous.Do

package object differentiable {

  /** @template */
  type INDArray = Do[Tape[org.nd4j.linalg.api.ndarray.INDArray, org.nd4j.linalg.api.ndarray.INDArray]]

  /** @template */
  type Float = Do[Tape[scala.Float, scala.Float]]

  /** @template */
  type Double = Do[Tape[scala.Double, scala.Double]]

  /** @template */
  type Any = Do[Tape[scala.Any, scala.Nothing]]
}
