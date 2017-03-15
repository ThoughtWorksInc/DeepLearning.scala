package com.thoughtworks.deeplearning

import java.nio.ByteBuffer

import com.qifun.statelessFuture.Future
import com.qifun.statelessFuture.Future.Stateless
import com.thoughtworks.deeplearning.Layer.{Batch, Tape}
import com.thoughtworks.deeplearning.Layer.Batch.Aux
import com.thoughtworks.deeplearning.OpenCL.Buffer
import com.thoughtworks.deeplearning.OpenCLCompiler.{Context, DslExpression, DslType, Kernel}
import org.lwjgl.BufferUtils
import org.lwjgl.opencl.CL10._
import shapeless.{Data0, _}
import shapeless.ops.hlist.{LiftAll, Mapper, RightFolder}
import com.dongxiguo.fastring.Fastring.Implicits._
import com.thoughtworks.deeplearning.OpenCL.NDRangeKernelEvent.Dimension
import com.thoughtworks.deeplearning.OpenCLCompiler.DslType.DslStructure

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableOpenCLBuffer {

  private[DifferentiableOpenCLBuffer] trait BufferBatch[ElementData, ElementDelta] extends Batch {
    override type Data = OpenCL.Buffer[ElementData]
    override type Delta = OpenCL.Buffer[ElementDelta]
  }

  trait KernelLayer {
    import KernelLayer._

    def upstreams: Seq[Upstream[Input, _, _]]
    def wengertVariables: Seq[MemoryType[_]]
//    def outputs: Seq[]

//    def forwardKernel

    type Input <: Tape
//    type OutputData = OpenCL.Buffer[ElementData]
//    type OutputDelta = OpenCL.Buffer[ElementDelta]
//    type ElementData
//    type ElementDelta
//
//    type Tape <: HList
//    type BufferTape <: HList
//
//    type Operands <: HList
//
//    def  upstreamDataType: DslStructure
//    def upstreamDeltaType: DslStructure
//
//    def operands: Operands
//
//    def upstreams(input: Input, operands: Operands): Upstream
//
//    def setUpstreamArgs(kernel: OpenCL.Kernel, i: Int, upstream: Upstream): Int
//
////    def upstreamValues: Mapper.Aux[Forward[Input], Upstream, UpstreamData]
////
////    def upstreamDataMemories: UpstreamDataMemory
//
//    // TODO: 证明Upvalues每一项都是 Layer.Aux[Input,...]
//    // TODO: 证明Upvalue每一项的OutputData、OutputDelta都支持Memory、而且支持DslType，而且可以组装成UpstreamData、CompactUpstreamDelta
//
//    type Upstream <: HList
//    type UpstreamData <: HList
//    type UpstreamDataMemory <: HList
//    type UpstreamDelta <: HList
//
//    def kernelForward: OpenCLCompiler.DslExpression // [UpstreamData, ElementData :: Tape]
//    def kernelBackward: OpenCLCompiler.DslExpression // [Buffer[ElementDelta] :: BufferTape, UpstreamDelta]
//
//    def elementDataMemory: Memory[ElementData]
//    def elementDataType: DslType
//    def elementDeltaType: DslType
//    def tapeType: DslStructure

  }

  object KernelLayer {

    trait Upstream[Input <: Tape, UpstreamOutputData, UpstreamOutputDelta] {
      def operand: Layer.Aux[Input, Tape.Aux[UpstreamOutputData, UpstreamOutputDelta]]
      def upstreamDataMemory: Memory[UpstreamOutputData]
      def upstreamDeltaMemory: Memory[UpstreamOutputDelta]
      def upstreamDataType: DslType
      def upstreamDeltaType: DslType
    }

    trait MemoryType[A] {
      def memory: Memory[A]
      def tapeType: DslType
    }

//    final class Forward[Input <: Batch](input: Input) extends Poly1
//
//    object DataTypeBuilder extends Poly2 {}
//    object DeltaTypeBuilder extends Poly2 {}
//
//    type Aux[Input0, ElementData0, ElementDelta0, UpstreamData0 <: HList, UpstreamDelta0 <: HList, Tape0 <: HList] =
//      KernelLayer {
//        type Input = Input0
//        type ElementData = ElementData0
//        type ElementDelta = ElementDelta0
//        type UpstreamData = UpstreamData0
//        type UpstreamDelta = UpstreamDelta0
//        type Tape = Tape0
//      }
//
//    object OutputDataMapper extends Poly1 {
//      // TODO: Use ToLayer instead
//      implicit def getOutputData[InputData, InputDelta, OutputData, OutputDelta] =
//        at[Layer.Aux[Batch.Aux[InputData, InputDelta], Batch.Aux[OutputData, OutputDelta]]] { _ =>
//          ??? : OutputData
//        }
//    }
  }

  // TODO: Merge kernel layers into normal layers
  object KernelLayers {

//    final case class FloatLiteral[InputData0, InputDelta0](reference: Float,
//                                                           inputDataType: DslType[InputData0],
//                                                           inputDeltaType: DslType[InputDelta0])
//        extends KernelLayer {
//      override type InputData = InputData0
//      override type InputDelta = InputDelta0
//      override type OutputData = Float
//      override type OutputDelta = Float
//      override type Tape = HNil
//      override type Upvalue = HNil
//
//      override def forward: DslExpression[InputData :: Upvalue, OutputData :: Tape] = {
//        OpenCLCompiler.DslExpression
//          .HCons(OpenCLCompiler.DslExpression.FloatLiteral(reference), OpenCLCompiler.DslExpression.HNilLiteral)
//      }
//
//      override def backward: DslExpression[OutputDelta :: Tape, HNil] = {
//        OpenCLCompiler.DslExpression.HNilLiteral
//      }
//
//      override def upvalueType: DslType.DslStructure.Nil.type = DslType.DslStructure.Nil
//
//      override def tapeType: DslType.DslStructure.Nil.type = DslType.DslStructure.Nil
//
//      override def outputDataType: DslType[Float] = DslType.DslFloat
//
//      override def outputDeltaType: DslType[Float] = DslType.DslFloat
//    }

  }

//
//  def floatLiteral[InputData0, InputDelta0](reference: Float)(implicit inputDataType: DslType[InputData0],
//                                                          inputDeltaType: DslType[InputDelta0]) = {
//    new KernelLayers.FloatLiteral(reference, inputDataType, inputDeltaType)
//  }

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

    final case class Fill[Input0 <: Batch,
                          ElementData,
                          ElementDelta,
                          UpstreamData <: HList,
                          UpstreamDelta <: HList,
                          Tape <: HList](
        clContext: OpenCL.Context,
        clCommandQueue: OpenCL.CommandQueue,
        size: Layer.Aux[Input0, Batch.Aux[Int, Float]],
        differentiableKernel: KernelLayer.Aux[Input0, ElementData, ElementDelta, UpstreamData, UpstreamDelta, Tape]
    ) extends Layer {
      override type Input = Input0

      trait Output extends BufferBatch[ElementData, ElementDelta]

      override def forward(input: Input): Future.Stateless[Output] = {
        Future {
          val sizeBatch = size.forward(input).await
          val clBuffer = clContext.createBuffer[ElementData](sizeBatch.value)(differentiableKernel.elementDataMemory)

          val forwardKernelTree =
            Kernel(
              name = "forward",
              numberOfDimensions = 1,
              dslFunction = differentiableKernel.kernelForward,
              inputType = differentiableKernel.upstreamDataType,
              outputType = DslType.dslHCons(differentiableKernel.elementDataType, differentiableKernel.tapeType)
            )

          val backwardKernelTree = Kernel(
            name = "backward",
            numberOfDimensions = 1,
            dslFunction = differentiableKernel.kernelBackward,
            inputType = DslType.dslHCons(differentiableKernel.elementDeltaType, differentiableKernel.tapeType),
            outputType = differentiableKernel.upstreamDeltaType
          )

          val forwardCode = OpenCLCompiler.generateSourceCode(forwardKernelTree, backwardKernelTree)
//          println(forwardCode.mkString)
          val program = clContext.createProgramWithSource(forwardCode)
          program.build().await
          val forwardKernel = program.createKernel("forward")

          val upstreams = differentiableKernel.upstreams(input, differentiableKernel.operands)

          val i = differentiableKernel.setUpstreamArgs(forwardKernel, 0, upstreams)
          val j = forwardKernel.setArg(i, clBuffer)

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
              // TODO: close operands
            }
          }
        }
      }
    }

  }

}
