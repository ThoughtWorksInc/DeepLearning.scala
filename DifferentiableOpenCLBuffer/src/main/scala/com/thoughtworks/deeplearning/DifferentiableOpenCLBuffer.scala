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
import com.thoughtworks.deeplearning.OpenCLCompiler.DslType.HListType

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableOpenCLBuffer {

  private[DifferentiableOpenCLBuffer] trait BufferBatch[ElementData, ElementDelta] extends Batch {
    override type Data = OpenCL.Buffer[ElementData]
    override type Delta = OpenCL.Buffer[ElementDelta]
  }

  trait KernelLayer {
    type InputData
    type InputDelta
    type OutputData
    type OutputDelta
//    type Upvalue <: HList
    type Upvalue <: HList // List of OpenCL.Buffer
    type Cache <: HList

    // TODO: upvalues

    def forward: OpenCLCompiler.DslFunction[InputData :: Upvalue, OutputData :: Cache]
    def backward: OpenCLCompiler.DslFunction[OutputDelta :: Cache, HNil]
    def inputDataType: DslType[InputData]
    def inputDeltaType: DslType[InputDelta]
    def outputDataType: DslType[OutputData]
    def outputDeltaType: DslType[OutputDelta]
    def upvalueType: HListType[Upvalue]
    def cacheType: HListType[Cache]

  }

  object KernelLayer {
    type Aux[InputData0, InputDelta0, OutputData0, OutputDelta0, Upvalue0 <: HList, Cache0 <: HList] =
      KernelLayer {
        type InputData = InputData0
        type InputDelta = InputDelta0
        type OutputData = OutputData0
        type OutputDelta = OutputDelta0
        type Upvalue = Upvalue0
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

    final case class FloatLiteral[InputData0, InputDelta0](value: Float,
                                                           inputDataType: DslType[InputData0],
                                                           inputDeltaType: DslType[InputDelta0])
        extends KernelLayer {
      override type InputData = InputData0
      override type InputDelta = InputDelta0
      override type OutputData = Float
      override type OutputDelta = Float
      override type Cache = HNil
      override type Upvalue = HNil

      override def forward: DslFunction[InputData :: Upvalue, OutputData :: Cache] = {
        OpenCLCompiler.DslFunction
          .HCons(OpenCLCompiler.DslFunction.FloatLiteral(value), OpenCLCompiler.DslFunction.HNilLiteral)
      }

      override def backward: DslFunction[OutputDelta :: Cache, HNil] = {
        OpenCLCompiler.DslFunction.HNilLiteral
      }

      override def upvalueType: DslType.DslHNil.type = DslType.DslHNil

      override def cacheType: DslType.DslHNil.type = DslType.DslHNil

      override def outputDataType: DslType[Float] = DslType.DslFloat

      override def outputDeltaType: DslType[Float] = DslType.DslFloat
    }

  }

  // TODO: closure and upvalue support

  def floatLiteral[InputData0, InputDelta0](value: Float)(implicit inputDataType: DslType[InputData0],
                                                          inputDeltaType: DslType[InputDelta0]) = {
    new KernelLayers.FloatLiteral(value, inputDataType, inputDeltaType)
  }

  object Layers {

    final case class Literal[InputData0, InputDelta0, OutputData0, OutputDelta0](value: OutputData0)
        extends Layer
        with Batch {
      override type Input = Batch.Aux[InputData0, InputDelta0]
      override type Output = Batch.Aux[OutputData0, OutputDelta0]
      override type Data = OutputData0
      override type Delta = OutputDelta0

      override def forward(input: Input) = Future(this)
      override def addReference() = this
      override protected def forceBackward(delta: Delta) = Future(())
      override def isTrainable: Boolean = false

      override def close(): Unit = {}
    }

    final case class Fill[InputData, InputDelta, ElementData: Memory, ElementDelta, Upvalue <: HList, Cache <: HList](
        clContext: OpenCL.Context,
        clCommandQueue: OpenCL.CommandQueue,
        size: Layer.Aux[Batch.Aux[InputData, InputDelta], Batch.Aux[Int, Float]],
        differentiableKernel: KernelLayer.Aux[InputData, InputDelta, ElementData, ElementDelta, Upvalue, Cache]
    ) extends Layer {
      override type Input = Batch.Aux[InputData, InputDelta]

      trait Output extends BufferBatch[ElementData, ElementDelta]

      override def forward(input: Input): Future.Stateless[Output] = {
        Future {
          val sizeBatch = size.forward(input).await
          val clBuffer = clContext.createBuffer[ElementData](sizeBatch.value)

          val forwardKernelTree =
            Kernel(
              name = "forward",
              numberOfDimensions = 1,
              dslFunction = differentiableKernel.forward,
              inputType = DslType.dslHCons(differentiableKernel.inputDataType, differentiableKernel.upvalueType),
              outputType = DslType.dslHCons(differentiableKernel.outputDataType, differentiableKernel.cacheType)
            )

          val backwardKernelTree = Kernel(
            name = "backward",
            numberOfDimensions = 1,
            dslFunction = differentiableKernel.backward,
            inputType = DslType.dslHCons(differentiableKernel.outputDeltaType, differentiableKernel.cacheType),
            outputType = DslType.DslHNil
          )

          val forwardCode = OpenCLCompiler.toSourceCode(forwardKernelTree, backwardKernelTree)
//          println(forwardCode.mkString)
          val program = clContext.createProgramWithSource(forwardCode)
          program.build().await
          val forwardKernel = program.createKernel("forward")
          forwardKernel.setArg(0, clBuffer)
          clCommandQueue
            .ndRangeKernel(forwardKernel, Seq(Dimension(0L, sizeBatch.value.toLong, sizeBatch.value.toLong)))
            .await

          // TODO: run kernel
          new Output {

            override def isTrainable = true

            override def value = {
              clBuffer
            }

            override def addReference() = ???

            override protected def forceBackward(delta: Buffer[ElementDelta]) = Future {}

            override def close() = {
              // TODO: close upvalues
            }
          }
        }
      }
    }

  }

}
