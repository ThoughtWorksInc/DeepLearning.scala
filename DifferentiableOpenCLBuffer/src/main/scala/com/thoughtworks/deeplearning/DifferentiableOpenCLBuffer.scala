package com.thoughtworks.deeplearning

import java.nio.ByteBuffer

import com.qifun.statelessFuture.Future
import com.qifun.statelessFuture.Future.Stateless
import com.thoughtworks.deeplearning.DifferentiableOpenCLBuffer.KernelLayer.OutputDataMapper
import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.Layer.Batch.Aux
import com.thoughtworks.deeplearning.OpenCL.Buffer
import com.thoughtworks.deeplearning.OpenCLCompiler.{Context, DslFunction, DslType, Kernel}
import org.lwjgl.BufferUtils
import org.lwjgl.opencl.CL10._
import shapeless.{Data0, _}
import shapeless.ops.hlist.Mapper
import com.dongxiguo.fastring.Fastring.Implicits._
import com.thoughtworks.deeplearning.OpenCL.NDRangeKernelEvent.Dimension

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableOpenCLBuffer {

  private[DifferentiableOpenCLBuffer] trait BufferBatch[ElementData, ElementDelta] extends Batch {
    override type Data = OpenCL.Buffer[ElementData]
    override type Delta = OpenCL.Buffer[ElementDelta]
  }

  trait KernelLayerCode {
    type InputData
    type OutputData
    type OutputDelta
    type InputDelta
    type Upvalues <: HList
    type Cache <: HList

    def forward: OpenCLCompiler.DslFunction[InputData :: Upvalues, OutputData :: Cache :: HNil]
    def backward: OpenCLCompiler.DslFunction[OutputDelta :: Cache :: HNil, InputDelta]

    // TODO: DslType

  }

  object KernelLayerCode {
    type Aux[InputData0, InputDelta0, OutputData0, OutputDelta0, Upvalues0 <: HList, Cache0 <: HList] =
      KernelLayerCode {
        type InputData = InputData0
        type InputDelta = InputDelta0
        type OutputData = OutputData0
        type OutputDelta = OutputDelta0
        type Upvalues = Upvalues0
        type Cache = Cache0
      }
  }

  trait KernelLayer {
    type InputData
    type InputDelta
    type OutputData
    type OutputDelta
    type UpvalueLayers <: HList
    type Cache <: HList

    def code(context: OpenCLCompiler.Context)(
        implicit outputDataMapper: shapeless.ops.hlist.Mapper[KernelLayer.OutputDataMapper.type, UpvalueLayers])
      : KernelLayerCode.Aux[InputData, InputDelta, OutputData, OutputDelta, outputDataMapper.Out, Cache]
  }

  object KernelLayer {
    type Aux[InputData0, InputDelta0, OutputData0, OutputDelta0, UpvalueLayers0 <: HList, Cache0 <: HList] =
      KernelLayer {
        type InputData = InputData0
        type InputDelta = InputDelta0
        type OutputData = OutputData0
        type OutputDelta = OutputDelta0
        type UpvalueLayers = UpvalueLayers0
        type Cache = Cache0
      }

    object OutputDataMapper extends Poly1 {
      // TODO: Use ToLayer instead
      implicit def getOutputData[InputData, InputDelta, OutputData, OutputDelta] =
        at[Layer.Aux[Batch.Aux[InputData, InputDelta], Batch.Aux[OutputData, OutputDelta]]] { _ =>
          ??? : OutputData
        }
    }
  }

  // TODO: Merge kernel layers into normal layers
  object KernelLayers {

    final case class Literal[Data0](override val value: Data0) extends KernelLayer with Layer with Batch {

      /**
        * Returns a new [[Batch]] that shares the same [[value]] and [[backward]] behavior with this [[Batch]].
        *
        * @note The newly created [[Batch]] and this [[Batch]] must be [[close]]d independently.
        */
      override def addReference(): Batch.Aux[Data0, Delta] = this
      override protected def forceBackward(delta: Delta): Stateless[Unit] = Future(())
      override def isTrainable: Boolean = ???

      override type Delta = Any
      override type InputData = Any
      override type InputDelta = Nothing

      override type Input = Batch
      override type Output = Batch.Aux[Data, Delta]

      override type OutputData = Data
      override type OutputDelta = Delta
      override type Data = Data0
      override type Cache = HNil
      override type UpvalueLayers = HNil
      override def code(context: Context)(implicit outputDataMapper: Mapper[OutputDataMapper.type, UpvalueLayers]) =
        ???

      override def forward(input: Input): Stateless[Output] = Future(this)

      override def close(): Unit = {}
    }

  }

  // TODO: closure and upvalue support

  object Layers {
    final case class Fill[InputData,
                          InputDelta,
                          ElementData: Memory,
                          ElementDelta,
                          UpvalueLayers <: HList,
                          Cache <: HList](
        clContext: OpenCL.Context,
        clCommandQueue: OpenCL.CommandQueue,
        size: Layer.Aux[Batch.Aux[InputData, InputDelta], Batch.Aux[Int, Float]],
        f: KernelLayer.Aux[InputData, InputDelta, ElementData, ElementDelta, UpvalueLayers, Cache] // TODO: replace f to a KernelLayer
    ) extends Layer {
      override type Input = Batch.Aux[InputData, InputDelta]

      trait Output extends BufferBatch[ElementData, ElementDelta]

      override def forward(input: Input): Future.Stateless[Output] = {
        Future {
          val sizeBatch = size.forward(input).await
          val clBuffer = clContext.createBuffer[ElementData](sizeBatch.value)

          val f = DslFunction.FloatLiteral(3.14f)
          val kernelTree = Kernel("f", 1, f, DslType.DslHNil, DslType.DslFloat)

          val code = OpenCLCompiler.toSourceCode(kernelTree)
          val program = clContext.createProgramWithSource(code)
          program.build().await
          val kernel = program.createKernel("f")
          kernel.setArg(0, clBuffer)
          clCommandQueue
            .ndRangeKernel(kernel, Seq(Dimension(0L, sizeBatch.value.toLong, sizeBatch.value.toLong)))
            .await

          // TODO: run kernel
          new Output {

            override def isTrainable = true

            override def value = {
              clBuffer
            }

            /**
              * Returns a new [[Batch]] that shares the same [[value]] and [[backward]] behavior with this [[Batch]].
              *
              * @note The newly created [[Batch]] and this [[Batch]] must be [[close]]d independently.
              */
            override def addReference() = ???

            override protected def forceBackward(delta: Buffer[ElementDelta]) = ???

            override def close() = ???
          }
        }
      }
    }

  }

}
