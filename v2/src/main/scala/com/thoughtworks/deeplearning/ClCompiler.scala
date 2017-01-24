package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Batch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object ClCompiler {
  object Layers {
    final case class Kernel[InputData, InputDelta, OutputData, OutputDelta]() extends BufferedLayer {

      type Input = Batch.Aux[InputData, InputDelta]

      type BufferedBatch = ReferenceCount {
        type Data = OutputData
        type Delta = OutputDelta
      }

      /**
        * Performs the underlying forward pass.
        *
        * @return a [[com.thoughtworks.deeplearning.Layer.Batch Batch]] that will be cached for subsequent [[forward]]
        */
      override protected def rawForward(input0: Input): BufferedBatch = {
//        new {
//          override protected final val input = input0
//        } with ReferenceCount {
//          type Data = OutputData
//          type Delta = OutputDelta
//
//          override protected def flush(): Unit = ???
//
//          override protected def closeUpstreams(): Unit = ???
//
//          override protected def forceBackward(delta: Delta): Unit = ???
//
//          override def isTrainable: Boolean = ???
//
//          override def value: Data = ???
//        }
        ???
      }
    }
  }
//def compile[Input, Output](layer: Layer[])
}
