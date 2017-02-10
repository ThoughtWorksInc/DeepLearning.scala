package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Batch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableOpenCLBuffer {

  private trait BufferBatch extends Batch {
    override type Data = OpenCL.Buffer
    override type Delta = OpenCL.Buffer
  }

  object Layers {

    final case class Fill[Input0 <: Batch]() extends Layer {
      override type Input = Input0
      trait Output extends BufferBatch

      override def forward(input: Input): Output = ???
    }

  }

}
