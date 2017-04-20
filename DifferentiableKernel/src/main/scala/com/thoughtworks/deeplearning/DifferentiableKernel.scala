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

//
//    def compile[OutputElementData, Jacobian <: HList, OutputElementDelta, ParameterRecord <: HList](
//        layer: OpenCLLayer[OpenCL.Buffer[OutputElementData], Jacobian],
//        context: OpenCL.Context,
//        device: Device,
//        commandQueue: CommandQueue)(implicit outputDataMemory: Memory[OutputElementData],
//                                    outputDeltaMemory: Memory[OutputElementDelta],
//                                    outputDataType: StaticDslType[OutputElementData],
//                                    outputDeltaType: StaticDslType[OutputElementDelta],
//                                    executor: ExecutionContext): RAIITask[(Int, ParameterRecord) => RAIITask[
//      Tape.Aux[OpenCL.Buffer[OutputElementData], OpenCL.Buffer[OutputElementDelta]]]] = {
//      import layer._
//      RAIITask.jump().each
//
//      val forwardOutputParameter = Parameter(OutputId, DslType.DslBuffer(outputDataType))
//
//      val (forwardParameters, inputSetter) = inputMetadataMap.view.zipWithIndex
//        .foldRight[(List[Parameter], Setter[OutputElementData])]((forwardOutputParameter :: Nil, EmptySetter)) {
//          case (((symbol, metadata), i), (accumulatedParameters, accumulatedSetter)) =>
//            val parameter = Parameter(symbol, metadata.dataType)
//            val setter: Setter[OutputElementData] = { (kernel, input) =>
//              val inputData = input(symbol).data
//              kernel.setArg[inputData.type](i, inputData)(
//                inputMetadataMap(symbol).dataMemory.asInstanceOf[Memory[inputData.type]])
//              accumulatedSetter(kernel, input)
//            }
//            // TODO: cache
//            (parameter :: accumulatedParameters, setter)
//        }
//
//      def forwardKernelDefinition: KernelDefinition = {
//        val outputIndex = {
//          // TODO: Support n-dimension Array
//          DslExpression.GetGlobalId(DslExpression.IntLiteral(0))
//        }
//        val effect = DslEffect.Update(DslExpression.Identifier(OutputId), outputIndex, data, outputDataType)
//        KernelDefinition(ForwardKernelName, forwardParameters, Seq(effect))
//      }
//
//      val forwardSource = OpenCLCodeGenerator.generateSourceCode(forwardKernelDefinition).toArray[CharSequence]
//      val forwordProgram = RAIITask.managed(context.createProgramWithSource(forwardSource)).each
//      RAIITask.unmanaged(forwordProgram.build()).each
//      val forwardKernelTask = RAIITask.managed(forwordProgram.createKernel(ForwardKernelName))
//
//      val backwardPrograms: Map[Any, RAIITask[Kernel]] = {
//        jacobian.view.zipWithIndex
//          .foldLeft(RAIITask.delay(Map.empty[Any, RAIITask[Kernel]])) {
//            case (accumulator, ((id, backwardExpression), index)) =>
//              throwableMonadic[RAIITask] {
//                val kernelName = backwardKernelName(index)
//                val backwardParameters = Seq(
//                  Parameter(OutputId, DslType.DslBuffer(outputDeltaType)),
//                  Parameter(id, inputMetadataMap(id).deltaType)
//                ) // TODO: cache
//                def backwardKernelDefinition: KernelDefinition = {
//
//                  val backwardIndex = {
//                    // TODO: Support n-dimension Array
//                    DslExpression.GetGlobalId(DslExpression.IntLiteral(0))
//                  }
//                  val effects = trainable.backward(id,
//                                                   inputMetadataMap(id).deltaType,
//                                                   backwardExpression,
//                                                   StaticDslExpression(DslExpression.Identifier(OutputId)))
//                  //                  val effect = DslEffect.Update(
//                  //                    DslExpression.Identifier(id),
//                  //                    backwardIndex,
//                  //                    DslExpression.Times(DslExpression.Identifier(OutputId), backwardExpression, outputDeltaType),
//                  //                    inputMetadataMap(id).deltaType
//                  //                  )
//                  KernelDefinition(kernelName, backwardParameters, effects)
//                }
//                val backwardSource =
//                  OpenCLCodeGenerator.generateSourceCode(backwardKernelDefinition).toArray[CharSequence]
//                val backwardProgram = RAIITask.managed(context.createProgramWithSource(backwardSource)).each
//                val backwardKernelTask = RAIITask.managed(forwordProgram.createKernel(kernelName))
//                accumulator.each.updated(id, backwardKernelTask)
//              }
//          }
//          .each
//      }
//
//      { (expectedSize: Int, inputParameterMap: Map[Any, Tape]) =>
//        throwableMonadic[RAIITask] {
//          val kernel = forwardKernelTask.each
//          val outputBuffer =
//            RAIITask.managed(context.createBuffer[OutputElementData](expectedSize)(outputDataMemory)).each
//          inputSetter(kernel, inputParameterMap)
//          kernel.setArg(inputMetadataMap.size, outputBuffer)
//          val event =
//            RAIITask
//              .managed(
//                commandQueue.enqueueNDRangeKernel(kernel, Seq(GlobalWorkSizeOnlyDimension(Address(expectedSize)))))
//              .each
//          RAIITask.unmanaged(event.waitForComplete()).each
//          new Tape {
//            override def data: OpenCL.Buffer[OutputElementData] = outputBuffer
//
//            private def deltaTask[Data0, Delta0, OutputDeltaBuffer <: OpenCL.Buffer[OutputElementDelta]](
//                outputDeltaTask: RAIITask[OutputDeltaBuffer],
//                backwardKernelTask: RAIITask[Kernel],
//                metadata: InputMetadata.Aux[Data0, Delta0],
//                inputTape: Tape.Aux[OpenCL.Buffer[Data0], OpenCL.Buffer[Delta0]]
//            ): Future[Unit] = {
//
//              val raiiTask = throwableMonadic[RAIITask] {
//                val kernel = backwardKernelTask.each
//                val outputDelta = outputDeltaTask.each
//                kernel.setArg(0, outputDelta: OpenCL.Buffer[OutputElementDelta])
//                val inputDeltaBuffer: OpenCL.Buffer[Delta0] =
//                  context.createBuffer(expectedSize)(metadata.deltaMemory)
//                // TODO: clEnqueueFillBuffer
//                try {
//                  kernel.setArg(1, inputDeltaBuffer)
//                  val event: OpenCL.Event =
//                    RAIITask
//                      .managed[OpenCL.Event](
//                        commandQueue.enqueueNDRangeKernel(kernel,
//                                                          Seq(GlobalWorkSizeOnlyDimension(Address(expectedSize)))))
//                      .each
//                  RAIITask.unmanaged(event.waitForComplete()).each
//                } catch {
//                  case e if NonFatal(e) =>
//                    inputDeltaBuffer.close()
//                    (throw e): Unit
//                }
//                inputDeltaBuffer
//              }.asInstanceOf[RAIITask[OpenCL.Buffer[Delta0]]] // FIXME: Avoid asInstanceOf
//
//              inputTape.backward[OpenCL.Buffer[Delta0]](
//                RAIITask.managed(RAIITask.run(raiiTask)): RAIITask[OpenCL.Buffer[Delta0]])
//            }
//
//            override def backward[OutputDeltaBuffer <: OpenCL.Buffer[OutputElementDelta]](
//                outputDeltaTask: RAIITask[OutputDeltaBuffer]): Future[Unit] = {
//              Future.suspend {
//
//                Parallel.unwrap((for ((id, tape) <- inputParameterMap) yield {
//                  val metadata = inputMetadataMap(id)
//                  Parallel(
//                    deltaTask(
//                      outputDeltaTask,
//                      backwardPrograms(id),
//                      metadata.self,
//                      inputParameterMap(id)
//                        .asInstanceOf[Tape.Aux[OpenCL.Buffer[metadata.Data], OpenCL.Buffer[metadata.Delta]]]
//                    ))
//                }).suml(Monoid.liftMonoid[ParallelFuture, Unit]))
//              }
//
//            }
//
//            // TODO: Change OutputData and OutputDelta to a pair of OpenCL.Buffer and OpenCL.Event
//            override type Data = OpenCL.Buffer[OutputElementData]
//            override type Delta = OpenCL.Buffer[OutputElementDelta]
//          }: Tape.Aux[OpenCL.Buffer[OutputElementData], OpenCL.Buffer[OutputElementDelta]]
//        }
//      }
//    }
//
//  }

//
//
//  trait InputMetadata {
//
//    type Data
//    type Delta
//
//    def dataMemory: Memory[Data]
//
//    def deltaMemory: Memory[Delta]
//
//    def dataType: StaticDslType[Data]
//
//    def deltaType: StaticDslType[Delta]
//
//    private[DifferentiableKernel] def self: InputMetadata.Aux[Data, Delta] = this
//  }
//  object InputMetadata {
//    type Aux[Data0, Delta0] = InputMetadata {
//      type Data = Data0
//      type Delta = Delta0
//    }
//    def apply[Data0: Memory: StaticDslType, Delta0: Memory: StaticDslType]: InputMetadata.Aux[Data0, Delta0] =
//      new InputMetadata {
//        override type Data = Data0
//        override type Delta = Delta0
//
//        override def dataMemory: Memory[Data0] = implicitly
//
//        override def deltaMemory: Memory[Delta0] = implicitly
//
//        override def dataType: StaticDslType[Data0] = implicitly
//
//        override def deltaType: StaticDslType[Delta0] = implicitly
//      }
//  }
//
//  trait Trainable[Jacobian] {
//    type OutputElementDelta
//    def backward(inputId: Any, // TODO: Give Input a type
//                 inputType: DslType,
//                 delta: Jacobian,
//                 outputElementDelta: StaticDslExpression[OutputElementDelta]): Seq[DslEffect]
//  }
//  object Trainable {
//    type Aux[Jacobian, OutputElementDelta0] = Trainable[Jacobian] {
//      type OutputElementDelta = OutputElementDelta0
//    }
//
//    implicit object FloatTrainable extends Trainable[StaticDslExpression[Float]] {
//      override type OutputElementDelta = Float
//
//      override def backward(inputId: Any,
//                            inputType: DslType,
//                            delta: StaticDslExpression[Float],
//                            outputElementDelta: StaticDslExpression[Float]): Seq[DslEffect] = {
//        Seq(
//          DslEffect.Update(
//            DslExpression.Identifier(inputId),
//            DslExpression.IntLiteral(0),
//            DslExpression.Times(outputElementDelta, delta, dslFloat),
//            inputType
//          ))
//      }
//    }
//  }
//
//  final case class OpenCLLayer[Data, Delta](
//      inputMetadataMap: Map[Any, InputMetadata], // TODO: replace Maps to shapeless records and create proxy via RecordArgs
//      data: Data,
//      jacobian: Map[Any, Delta]
//  )
//  object OpenCLLayer {
//
//    type ForwardAccumulation[OutputElementData, OutputElementDelta] =
//      OpenCLLayer[StaticDslExpression[OutputElementData], StaticDslExpression[OutputElementDelta]]

//
//    def intLiteral(data: Int): OpenCLLayer.ForwardAccumulation[Int, Float] =
//      new OpenCLLayer.ForwardAccumulation[Int, Float](
//        Map.empty,
//        StaticDslExpression(DslExpression.IntLiteral(data)),
//        Map.empty
//      )
//
//    def getGlobalId(
//        operand: OpenCLLayer.ForwardAccumulation[Int, Float]): OpenCLLayer.ForwardAccumulation[Int, Float] =
//      new OpenCLLayer.ForwardAccumulation[Int, Float](
//        operand.inputMetadataMap,
//        StaticDslExpression[Int](DslExpression.GetGlobalId(operand.data)),
//        Map.empty
//      )
//
//    def identifier[Data: Memory: StaticDslType, Delta: Memory: StaticDslType](
//        id: Any) /*(implicit multiplicationMonoid: Monoid[StaticDslExpression[Delta] @@ Multiplication])*/ =
//      new OpenCLLayer.ForwardAccumulation[Data, Delta](
//        Map(id -> InputMetadata[Data, Delta]),
//        StaticDslExpression(DslExpression.Identifier(id)),
//        Map(id -> ??? /*Multiplication.unwrap(multiplicationMonoid.zero)*/ ) // TODO: delta for entire buffer
//      )
//
//    private[OpenCLLayer] final val OutputId = new AnyRef
//    @inline private[OpenCLLayer] final val ForwardKernelName = "forward"
//    @inline private[OpenCLLayer] final def backwardKernelName(index: Int) = raw"""backward_$index"""
//    private[OpenCLLayer] type Setter[-OutputData] = (Kernel, Map[Any, Tape]) => Unit
//    private[OpenCLLayer] val EmptySetter: Setter[Any] = (kernel, input) => ()
//
//    import OpenCLLayer._
//
//    /**
//      * @param executor on which the compilation runs
//      */
//    def compile[OutputElementData, Jacobian, OutputElementDelta](
//        layer: OpenCLLayer[StaticDslExpression[OutputElementData], Jacobian],
//        context: OpenCL.Context,
//        device: Device,
//        commandQueue: CommandQueue)(implicit trainable: Trainable.Aux[Jacobian, OutputElementDelta],
//                                    outputDataMemory: Memory[OutputElementData],
//                                    outputDeltaMemory: Memory[OutputElementDelta],
//                                    outputDataType: StaticDslType[OutputElementData],
//                                    outputDeltaType: StaticDslType[OutputElementDelta],
//                                    executor: ExecutionContext): RAIITask[(Int, Map[Any, Tape]) => RAIITask[
//      Tape.Aux[OpenCL.Buffer[OutputElementData], OpenCL.Buffer[OutputElementDelta]]]] = throwableMonadic[RAIITask] {
//      import layer._
//      RAIITask.jump().each
//
//      val forwardOutputParameter = Parameter(OutputId, DslType.DslBuffer(outputDataType))
//
//      val (forwardParameters, inputSetter) = inputMetadataMap.view.zipWithIndex
//        .foldRight[(List[Parameter], Setter[OutputElementData])]((forwardOutputParameter :: Nil, EmptySetter)) {
//          case (((symbol, metadata), i), (accumulatedParameters, accumulatedSetter)) =>
//            val parameter = Parameter(symbol, metadata.dataType)
//            val setter: Setter[OutputElementData] = { (kernel, input) =>
//              val inputData = input(symbol)
//              kernel.setArg[inputData.type](i, inputData)(
//                inputMetadataMap(symbol).dataMemory.asInstanceOf[Memory[inputData.type]])
//              accumulatedSetter(kernel, input)
//            }
//            // TODO: cache
//            (parameter :: accumulatedParameters, setter)
//        }
//
//      def forwardKernelDefinition: KernelDefinition = {
//        val outputIndex = {
//          // TODO: Support n-dimension Array
//          DslExpression.GetGlobalId(DslExpression.IntLiteral(0))
//        }
//        val effect = DslEffect.Update(DslExpression.Identifier(OutputId), outputIndex, data, outputDataType)
//        KernelDefinition(ForwardKernelName, forwardParameters, Seq(effect))
//      }
//
//      val forwardSource = OpenCLCodeGenerator.generateSourceCode(forwardKernelDefinition).toArray[CharSequence]
//      val forwordProgram = RAIITask.managed(context.createProgramWithSource(forwardSource)).each
//      RAIITask.unmanaged(forwordProgram.build()).each
//      val forwardKernelTask = RAIITask.managed(forwordProgram.createKernel(ForwardKernelName))
//
//      val backwardPrograms: Map[Any, RAIITask[Kernel]] = {
//        jacobian.view.zipWithIndex
//          .foldLeft(RAIITask.delay(Map.empty[Any, RAIITask[Kernel]])) {
//            case (accumulator, ((id, backwardExpression), index)) =>
//              throwableMonadic[RAIITask] {
//                val kernelName = backwardKernelName(index)
//                val backwardParameters = Seq(
//                  Parameter(OutputId, DslType.DslBuffer(outputDeltaType)),
//                  Parameter(id, inputMetadataMap(id).deltaType)
//                ) // TODO: cache
//                def backwardKernelDefinition: KernelDefinition = {
//
//                  val backwardIndex = {
//                    // TODO: Support n-dimension Array
//                    DslExpression.GetGlobalId(DslExpression.IntLiteral(0))
//                  }
//                  val effects = trainable.backward(id,
//                                                   inputMetadataMap(id).deltaType,
//                                                   backwardExpression,
//                                                   StaticDslExpression(DslExpression.Identifier(OutputId)))
////                  val effect = DslEffect.Update(
////                    DslExpression.Identifier(id),
////                    backwardIndex,
////                    DslExpression.Times(DslExpression.Identifier(OutputId), backwardExpression, outputDeltaType),
////                    inputMetadataMap(id).deltaType
////                  )
//                  KernelDefinition(kernelName, backwardParameters, effects)
//                }
//                val backwardSource =
//                  OpenCLCodeGenerator.generateSourceCode(backwardKernelDefinition).toArray[CharSequence]
//                val backwardProgram = RAIITask.managed(context.createProgramWithSource(backwardSource)).each
//                val backwardKernelTask = RAIITask.managed(forwordProgram.createKernel(kernelName))
//                accumulator.each.updated(id, backwardKernelTask)
//              }
//          }
//          .each
//      }
//
//      { (expectedSize: Int, inputParameterMap: Map[Any, Tape]) =>
//        throwableMonadic[RAIITask] {
//          val kernel = forwardKernelTask.each
//          val outputBuffer =
//            RAIITask.managed(context.createBuffer[OutputElementData](expectedSize)(outputDataMemory)).each
//          inputSetter(kernel, inputParameterMap)
//          kernel.setArg(inputMetadataMap.size, outputBuffer)
//          val event =
//            RAIITask
//              .managed(
//                commandQueue.enqueueNDRangeKernel(kernel, Seq(GlobalWorkSizeOnlyDimension(Address(expectedSize)))))
//              .each
//          RAIITask.unmanaged(event.waitForComplete()).each
//          new Tape {
//            override def data: OpenCL.Buffer[OutputElementData] = outputBuffer
//
//            private def deltaTask[Data0, Delta0, OutputDeltaBuffer <: OpenCL.Buffer[OutputElementDelta]](
//                outputDeltaTask: RAIITask[OutputDeltaBuffer],
//                backwardKernelTask: RAIITask[Kernel],
//                metadata: InputMetadata.Aux[Data0, Delta0],
//                inputTape: Tape.Aux[OpenCL.Buffer[Data0], OpenCL.Buffer[Delta0]]
//            ): Future[Unit] = {
//
//              val raiiTask = throwableMonadic[RAIITask] {
//                val kernel = backwardKernelTask.each
//                val outputDelta = outputDeltaTask.each
//                kernel.setArg(0, outputDelta: OpenCL.Buffer[OutputElementDelta])
//                val inputDeltaBuffer: OpenCL.Buffer[Delta0] =
//                  context.createBuffer(expectedSize)(metadata.deltaMemory)
//                // TODO: clEnqueueFillBuffer
//                try {
//                  kernel.setArg(1, inputDeltaBuffer)
//                  val event: OpenCL.Event =
//                    RAIITask
//                      .managed[OpenCL.Event](
//                        commandQueue.enqueueNDRangeKernel(kernel,
//                                                          Seq(GlobalWorkSizeOnlyDimension(Address(expectedSize)))))
//                      .each
//                  RAIITask.unmanaged(event.waitForComplete()).each
//                } catch {
//                  case e if NonFatal(e) =>
//                    inputDeltaBuffer.close()
//                    (throw e): Unit
//                }
//                inputDeltaBuffer
//              }.asInstanceOf[RAIITask[OpenCL.Buffer[Delta0]]] // FIXME: Avoid asInstanceOf
//
//              inputTape.backward[OpenCL.Buffer[Delta0]](
//                RAIITask.managed(RAIITask.run(raiiTask)): RAIITask[OpenCL.Buffer[Delta0]])
//            }
//
//            override def backward[OutputDeltaBuffer <: OpenCL.Buffer[OutputElementDelta]](
//                outputDeltaTask: RAIITask[OutputDeltaBuffer]): Future[Unit] = {
//              Future.suspend {
//
//                Parallel.unwrap((for ((id, tape) <- inputParameterMap) yield {
//                  val metadata = inputMetadataMap(id)
//                  Parallel(deltaTask(
//                    outputDeltaTask,
//                    backwardPrograms(id),
//                    metadata.self,
//                    inputParameterMap(id)
//                      .asInstanceOf[Tape.Aux[OpenCL.Buffer[metadata.Data], OpenCL.Buffer[metadata.Delta]]]
//                  ))
//                }).suml(Monoid.liftMonoid[ParallelFuture, Unit]))
//              }
//
//            }
//
//            // TODO: Change OutputData and OutputDelta to a pair of OpenCL.Buffer and OpenCL.Event
//            override type Data = OpenCL.Buffer[OutputElementData]
//            override type Delta = OpenCL.Buffer[OutputElementDelta]
//          }: Tape.Aux[OpenCL.Buffer[OutputElementData], OpenCL.Buffer[OutputElementDelta]]
//        }
//      }
//    }
//
