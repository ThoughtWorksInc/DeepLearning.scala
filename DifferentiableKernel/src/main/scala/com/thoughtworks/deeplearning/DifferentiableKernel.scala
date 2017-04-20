package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Memory.Address
import com.thoughtworks.deeplearning.OpenCL.CommandQueue.GlobalWorkSizeOnlyDimension
import com.thoughtworks.deeplearning.OpenCL.{CommandQueue, Device, Kernel}
import com.thoughtworks.deeplearning.OpenCLCodeGenerator.DslType.{DslBuffer, DslDouble, DslFloat, DslInt}
import com.thoughtworks.deeplearning.OpenCLCodeGenerator._
import com.thoughtworks.each.Monadic._
import com.thoughtworks.raii.RAIITask
import jdk.nashorn.internal.objects.annotations.Setter
import shapeless.labelled.FieldType
import shapeless._

import scala.concurrent.ExecutionContext
import scala.util.control.NonFatal
import scalaz.{@@, Monad, Monoid}
import scalaz.Tags.{Multiplication, Parallel}
import scalaz.concurrent.Future
import scalaz.concurrent.Future.{ParallelFuture, futureParallelApplicativeInstance}
import scalaz.std.anyVal._
import scalaz.std.iterable._
import scalaz.syntax.foldable._
import scala.language.higherKinds

object DifferentiableKernel {

  private[DifferentiableKernel] trait StaticDslTypeExtractor {
    type AbstractType[A] <: DslType

    implicit def dslDouble: AbstractType[Double]
    implicit def dslFloat: AbstractType[Float]
    implicit def dslInt: AbstractType[Int]
    implicit def dslBuffer[Element: AbstractType]: AbstractType[OpenCL.Buffer[Element]]
  }

  private[DifferentiableKernel] trait StaticDslExpressionExtractor {
    type AbstractType[A] <: DslExpression

    def apply[A](expression: DslExpression): AbstractType[A]
  }

  @inline val StaticDslExpression: StaticDslExpressionExtractor =
    new StaticDslExpressionExtractor {
      override type AbstractType[A] = DslExpression

      override def apply[A](expression: DslExpression): DslExpression = expression
    }

  @inline val StaticDslType: StaticDslTypeExtractor =
    new StaticDslTypeExtractor {
      @inline
      override final def dslDouble = DslDouble

      @inline
      override final def dslFloat = DslFloat

      @inline
      override final def dslInt = DslInt

      override type AbstractType[A] = DslType

      override def dslBuffer[Element](implicit elementType: DslType): DslType = DslBuffer(elementType)
    }
  // TODO: https://github.com/ClaireNeveu/macrame/issues/7
  type StaticDslType[A] = StaticDslType.AbstractType[A]
  type StaticDslExpression[A] = StaticDslExpression.AbstractType[A]

  import StaticDslType._

  final case class OpenCLLayer[OutputElementData, OutputElementDelta, Jacobian <: HList](
      data: StaticDslExpression[OutputElementData],
      jacobian: Jacobian) {

    import OpenCLLayer._

    def compile(context: OpenCL.Context, device: Device, commandQueue: CommandQueue)(
        implicit compiler: Compiler[OutputElementData, OutputElementDelta, Jacobian],
        outputDataMemory: Memory[OutputElementData],
        outputDeltaMemory: Memory[OutputElementDelta],
        outputDataType: StaticDslType[OutputElementData],
        outputDeltaType: StaticDslType[OutputElementDelta],
        executor: ExecutionContext): RAIITask[(Int, compiler.ParameterRecord) => RAIITask[
      Tape.Aux[OpenCL.Buffer[OutputElementData], OpenCL.Buffer[OutputElementDelta]]]] = throwableMonadic[RAIITask] {

      RAIITask.jump().each

      def forwardKernelDefinition: KernelDefinition = {
        val outputIndex = {
          // TODO: Support n-dimension Array
          DslExpression.GetGlobalId(DslExpression.IntLiteral(0))
        }
        val effect = DslEffect.Update(DslExpression.Identifier(OutputId), outputIndex, data, outputDataType)
        KernelDefinition(ForwardKernelName, compiler.forwardParameters, Seq(effect))
      }

      val forwardSource = OpenCLCodeGenerator.generateSourceCode(forwardKernelDefinition).toArray[CharSequence]
      val forwordProgram = RAIITask.managed(context.createProgramWithSource(forwardSource)).each
      RAIITask.unmanaged(forwordProgram.build()).each
      val forwardKernelTask = RAIITask.managed(forwordProgram.createKernel(ForwardKernelName))

      { (expectedSize: Int, inputParameterMap: compiler.ParameterRecord) =>
        throwableMonadic[RAIITask] {
          val kernel = forwardKernelTask.each
          val outputBuffer =
            RAIITask.managed(context.createBuffer[OutputElementData](expectedSize)(outputDataMemory)).each
          compiler.setKernelInputArguments(kernel, 1, inputParameterMap)
          kernel.setArg(0, outputBuffer)
          val event =
            RAIITask
              .managed(
                commandQueue.enqueueNDRangeKernel(kernel, Seq(GlobalWorkSizeOnlyDimension(Address(expectedSize)))))
              .each
          RAIITask.unmanaged(event.waitForComplete()).each
          new Tape {
            override def data: OpenCL.Buffer[OutputElementData] = outputBuffer

            override def backward[OutputDeltaBuffer <: OpenCL.Buffer[OutputElementDelta]](
                outputDeltaTask: RAIITask[OutputDeltaBuffer]): Future[Unit] = {
              Future.suspend {
                Future.now(()) // TODO: backward
              }

            }

            // TODO: Change OutputData and OutputDelta to a pair of OpenCL.Buffer and OpenCL.Event
            override type Data = OpenCL.Buffer[OutputElementData]
            override type Delta = OpenCL.Buffer[OutputElementDelta]
          }: Tape.Aux[OpenCL.Buffer[OutputElementData], OpenCL.Buffer[OutputElementDelta]]
        }
      }
    }
  }

  object OpenCLLayer {
    private[deeplearning] final val OutputId = new AnyRef
    @inline private[deeplearning] final val ForwardKernelName = "forward"
    @inline private[deeplearning] final def backwardKernelName(index: Int) = raw"""backward_$index"""

    def floatLiteral(data: Float): OpenCLLayer[Float, Float, HNil] =
      OpenCLLayer[Float, Float, HNil](
        StaticDslExpression[Float](DslExpression.FloatLiteral(data)),
        HNil
      )
  }

  import OpenCLLayer._

  trait InputCompiler[JacobianCell] {
    type Input <: Tape

    def forwardParameter: Parameter

    def setArgument(kernel: Kernel, startIndex: Int, input: Input): Unit
  }

  object InputCompiler {

    type Aux[JacobianCell, Input0] = InputCompiler[JacobianCell] {
      type Input = Input0
    }

  }

  trait Compiler[OutputElementData, OutputElementDelta, Jacobian <: HList] {
    type ParameterRecord <: HList

    def forwardInputParameters: List[Parameter]

    def forwardParameters(implicit outputDataType: StaticDslType[OutputElementData]): List[Parameter] =
      Parameter(OutputId, DslType.DslBuffer(outputDataType)) :: forwardInputParameters

    def setKernelInputArguments(kernel: Kernel, startIndex: Int, parameters: ParameterRecord)
  }

  object Compiler {
    type Aux[OutputElementData, OutputElementDelta, Jacobian <: HList, ParameterRecord0] =
      Compiler[OutputElementData, OutputElementDelta, Jacobian] {
        type ParameterRecord = ParameterRecord0
      }

    implicit def hnilFill[OutputElementData, OutputElementDelta]
      : Compiler.Aux[OutputElementData, OutputElementDelta, HNil, HNil] =
      new Compiler[OutputElementData, OutputElementDelta, HNil] {
        override type ParameterRecord = HNil

        override def forwardInputParameters: Nil.type = Nil

        override def setKernelInputArguments(kernel: Kernel, startIndex: Int, parameters: HNil): Unit = {}
      }

    implicit def hconsFill[OutputElementData,
                           OutputElementDelta0,
                           Key,
                           JacobianHead,
                           JacobianTail <: HList,
                           Input,
                           TailParameterRecord <: HList](
        implicit headInputCompiler: InputCompiler.Aux[JacobianHead, Input],
        tailCompiler: Compiler.Aux[OutputElementData, OutputElementDelta0, JacobianTail, TailParameterRecord])
      : Compiler.Aux[OutputElementData,
                     OutputElementDelta0,
                     FieldType[Key, JacobianHead] :: JacobianTail,
                     FieldType[Key, Input] :: TailParameterRecord] =
      new Compiler[OutputElementData, OutputElementDelta0, FieldType[Key, JacobianHead] :: JacobianTail] {
        override type ParameterRecord = FieldType[Key, Input] :: TailParameterRecord
        override def forwardInputParameters: List[Parameter] =
          headInputCompiler.forwardParameter :: tailCompiler.forwardInputParameters

        override def setKernelInputArguments(kernel: Kernel,
                                             startIndex: Int,
                                             parameters: FieldType[Key, Input] :: TailParameterRecord): Unit = {
          headInputCompiler.setArgument(kernel, startIndex, parameters.head)
          tailCompiler.setKernelInputArguments(kernel, startIndex + 1, parameters.tail)
        }
      }
  }

}
