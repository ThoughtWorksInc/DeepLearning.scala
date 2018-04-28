package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.compute.OpenCL
import com.thoughtworks.deeplearning.DeepLearning
import org.apache.commons.math3.linear.RealMatrix

/**
  * @author 杨博 (Yang Bo)
  */
trait Tensors extends OpenCL {

  final case class Tensor[Buffer](buffer: Buffer, shape: Array[Int], view: RealMatrix)

}
