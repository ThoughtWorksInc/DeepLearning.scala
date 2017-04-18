package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Memory.Address
import com.thoughtworks.deeplearning.OpenCL.CommandQueue.GlobalWorkSizeOnlyDimension
import com.thoughtworks.deeplearning.OpenCL.{CommandQueue, Device, Kernel}
import com.thoughtworks.deeplearning.OpenCLCodeGenerator._
import com.thoughtworks.each.Monadic._
import com.thoughtworks.raii.RAIITask

import scala.concurrent.ExecutionContext
import scala.util.control.NonFatal
import scalaz.Monoid
import scalaz.Tags.Parallel
import scalaz.concurrent.Future
import scalaz.concurrent.Future.{ParallelFuture, futureParallelApplicativeInstance}
import scalaz.std.anyVal._
import scalaz.std.iterable._
import scalaz.syntax.foldable._

object DifferentiableKernel {

  trait InputMetadata {

    type Data
    type Delta

    def dataMemory: Memory[Data]

    def deltaMemory: Memory[Delta]

    def dataType: DslType

    def deltaType: DslType

    private[DifferentiableKernel] def self: InputMetadata.Aux[Data, Delta] = this
  }
  object InputMetadata {
    type Aux[Data0, Delta0] = InputMetadata {
      type Data = Data0
      type Delta = Delta0
    }
  }
  object OpenCLLayer {

    final case class Reciprocal[OutputElementData0, OutputElementDelta0](
        operand0: OpenCLLayer
    ) extends OpenCLLayer {
      override type OutputElementDelta = OutputElementDelta0
      override type OutputElementData = OutputElementData0

      override def inputMetadataMap: Map[Any, InputMetadata] = operand0.inputMetadataMap

      override def outputDataMemory: Memory[OutputElementData0] = ???

      override def outputDeltaMemory: Memory[OutputElementDelta0] = ???

      override def outputDataType: DslType = ???

      override def outputDeltaType: DslType = ???

      override def forward: DslExpression = ???

      override def backward(outputDelta: DslExpression): Map[Any, DslExpression] = {
        //operand0.backward(upstreamDelta(outputDelta))
        ???
      }
    }

    final case class FloatLiteral(data: Float) extends OpenCLLayer {

      override type OutputElementDelta = Float
      override type OutputElementData = Float

      override def inputMetadataMap: Map[Any, InputMetadata] = Map.empty

      override def outputDataMemory: Memory[OutputElementData] = implicitly

      override def outputDeltaMemory: Memory[OutputElementDelta] = implicitly

      override def outputDataType: DslType = DslType.DslFloat

      override def outputDeltaType: DslType = DslType.DslFloat

      override def forward: DslExpression = {
        DslExpression.FloatLiteral(data)
      }

      override def backward(outputDelta: DslExpression): Map[Any, DslExpression] = Map.empty
    }

    private[OpenCLLayer] final val OutputDataId = new AnyRef
    private[OpenCLLayer] final val OutputDeltaId = new AnyRef
    @inline private[OpenCLLayer] final val ForwardKernelName = "forward"
    @inline private[OpenCLLayer] final def backwardKernelName(index: Int) = raw"""backward_$index"""
    private[OpenCLLayer] type Setter[-OutputData] = (Kernel, Map[Any, Tape]) => Unit
    private[OpenCLLayer] val EmptySetter: Setter[Any] = (kernel, input) => ()

  }

  trait OpenCLLayer {
    import OpenCLLayer._

    def inputMetadataMap: Map[Any, InputMetadata]

    type OutputElementData
    type OutputElementDelta

    def outputDataMemory: Memory[OutputElementData]

    def outputDeltaMemory: Memory[OutputElementDelta]

    def outputDataType: DslType

    def outputDeltaType: DslType

//    type CacheExpression = Seq[DslExpression]
    def forward: DslExpression

    def backward(outputDelta: DslExpression /*, cache: Seq[DslExpression]*/ ): Map[Any, DslExpression]

    /**
      * @param executor on which the compilation runs
      */
    def compile(context: OpenCL.Context, device: Device, commandQueue: CommandQueue)(
        implicit executor: ExecutionContext): RAIITask[(Int, Map[Any, Tape]) => RAIITask[
      Tape.Aux[OpenCL.Buffer[OutputElementData], OpenCL.Buffer[OutputElementDelta]]]] = throwableMonadic[RAIITask] {

      RAIITask.jump().each

      val forwardOutputParameter = Parameter(OutputDataId, DslType.DslBuffer(outputDataType))

      val (forwardParameters, inputSetter) = inputMetadataMap.view.zipWithIndex
        .foldRight[(List[Parameter], Setter[OutputElementData])]((forwardOutputParameter :: Nil, EmptySetter)) {
          case (((symbol, metadata), i), (accumulatedParameters, accumulatedSetter)) =>
            val parameter = Parameter(symbol, metadata.dataType)
            val setter: Setter[OutputElementData] = { (kernel, input) =>
              val inputData = input(symbol)
              kernel.setArg[inputData.type](i, inputData)(
                inputMetadataMap(symbol).dataMemory.asInstanceOf[Memory[inputData.type]])
              accumulatedSetter(kernel, input)
            }
            // TODO: cache
            (parameter :: accumulatedParameters, setter)
        }

      def forwardKernelDefinition: KernelDefinition = {
        val outputIndex = {
          // TODO: Support n-dimension Array
          DslExpression.GetGlobalId(DslExpression.IntLiteral(0))
        }
        val effect = DslEffect.Update(DslExpression.Identifier(OutputDataId), outputIndex, forward, outputDataType)
        KernelDefinition(ForwardKernelName, forwardParameters, Seq(effect))
      }

      val forwardSource = OpenCLCodeGenerator.generateSourceCode(forwardKernelDefinition).toArray[CharSequence]
      val forwordProgram = RAIITask.managed(context.createProgramWithSource(forwardSource)).each
      RAIITask.unmanaged(forwordProgram.build()).each
      val forwardKernelTask = RAIITask.managed(forwordProgram.createKernel(ForwardKernelName))

      val backwardPrograms: Map[Any, RAIITask[Kernel]] = {
        backward(DslExpression.Identifier(OutputDeltaId)).view.zipWithIndex
          .foldLeft(RAIITask.delay(Map.empty[Any, RAIITask[Kernel]])) {
            case (accumulator, ((id, backwardExpression), index)) =>
              throwableMonadic[RAIITask] {
                val kernelName = backwardKernelName(index)
                val backwardParameters = Seq(
                  Parameter(OutputDeltaId, DslType.DslBuffer(outputDeltaType)),
                  Parameter(id, inputMetadataMap(id).deltaType)
                ) // TODO: cache
                def backwardKernelDefinition: KernelDefinition = {

                  val backwardIndex = {
                    // TODO: Support n-dimension Array
                    DslExpression.GetGlobalId(DslExpression.IntLiteral(0))
                  }
                  val effect = DslEffect.Update(DslExpression.Identifier(id),
                                                backwardIndex,
                                                backwardExpression,
                                                inputMetadataMap(id).deltaType)
                  KernelDefinition(kernelName, backwardParameters, Seq(effect))
                }
                val backwardSource =
                  OpenCLCodeGenerator.generateSourceCode(backwardKernelDefinition).toArray[CharSequence]
                val backwardProgram = RAIITask.managed(context.createProgramWithSource(backwardSource)).each
                val backwardKernelTask = RAIITask.managed(forwordProgram.createKernel(kernelName))
                accumulator.each.updated(id, backwardKernelTask)
              }
          }
          .each
      }

      { (expectedSize: Int, inputParameterMap: Map[Any, Tape]) =>
        throwableMonadic[RAIITask] {
          val kernel = forwardKernelTask.each
          val outputBuffer =
            RAIITask.managed(context.createBuffer[OutputElementData](expectedSize)(outputDataMemory)).each
          inputSetter(kernel, inputParameterMap)
          kernel.setArg(inputMetadataMap.size, outputBuffer)
          val event =
            RAIITask
              .managed(
                commandQueue.enqueueNDRangeKernel(kernel, Seq(GlobalWorkSizeOnlyDimension(Address(expectedSize)))))
              .each
          RAIITask.unmanaged(event.waitForComplete()).each
          new Tape {
            override def data: OpenCL.Buffer[OutputElementData] = outputBuffer

            private def deltaTask[Data0, Delta0, OutputDeltaBuffer <: OpenCL.Buffer[OutputElementDelta]](
                outputDeltaTask: RAIITask[OutputDeltaBuffer],
                backwardKernelTask: RAIITask[Kernel],
                metadata: InputMetadata.Aux[Data0, Delta0],
                inputTape: Tape.Aux[OpenCL.Buffer[Data0], OpenCL.Buffer[Delta0]]
            ): Future[Unit] = {

              val raiiTask = throwableMonadic[RAIITask] {
                val kernel = backwardKernelTask.each
                val outputDelta = outputDeltaTask.each
                kernel.setArg(0, outputDelta: OpenCL.Buffer[OutputElementDelta])
                val inputDeltaBuffer: OpenCL.Buffer[Delta0] =
                  context.createBuffer(expectedSize)(metadata.deltaMemory)

                try {
                  kernel.setArg(1, inputDeltaBuffer)
                  val event: OpenCL.Event =
                    RAIITask
                      .managed[OpenCL.Event](
                        commandQueue.enqueueNDRangeKernel(kernel,
                                                          Seq(GlobalWorkSizeOnlyDimension(Address(expectedSize)))))
                      .each
                  RAIITask.unmanaged(event.waitForComplete()).each
                } catch {
                  case e if NonFatal(e) =>
                    inputDeltaBuffer.close()
                    (throw e): Unit
                }
                inputDeltaBuffer
              }.asInstanceOf[RAIITask[OpenCL.Buffer[Delta0]]] // FIXME: Avoid asInstanceOf

              inputTape.backward[OpenCL.Buffer[Delta0]](
                RAIITask.managed(RAIITask.run(raiiTask)): RAIITask[OpenCL.Buffer[Delta0]])
            }

            override def backward[OutputDeltaBuffer <: OpenCL.Buffer[OutputElementDelta]](
                outputDeltaTask: RAIITask[OutputDeltaBuffer]): Future[Unit] = {
              Future.suspend {

                Parallel.unwrap((for ((id, tape) <- inputParameterMap) yield {
                  val metadata = inputMetadataMap(id)
                  Parallel(deltaTask(
                    outputDeltaTask,
                    backwardPrograms(id),
                    metadata.self,
                    inputParameterMap(id)
                      .asInstanceOf[Tape.Aux[OpenCL.Buffer[metadata.Data], OpenCL.Buffer[metadata.Delta]]]
                  ))
                }).suml(Monoid.liftMonoid[ParallelFuture, Unit]))
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

}
