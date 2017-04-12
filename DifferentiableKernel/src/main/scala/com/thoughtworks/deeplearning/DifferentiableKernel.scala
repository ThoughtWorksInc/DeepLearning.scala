package com.thoughtworks.deeplearning

import java.util.concurrent.{Executor, ExecutorService}

import com.thoughtworks.deeplearning.DifferentiableKernel._
import com.thoughtworks.deeplearning.Memory.Address
import com.thoughtworks.deeplearning.OpenCL.CommandQueue.GlobalWorkSizeOnlyDimension
import com.thoughtworks.deeplearning.OpenCL.{CommandQueue, Device, Kernel, Program}
import com.thoughtworks.deeplearning.OpenCLCodeGenerator.DslExpression.GetGlobalId
import com.thoughtworks.deeplearning.OpenCLCodeGenerator._
import com.thoughtworks.raii.RAIITask

import scalaz.\/-
import scalaz.concurrent.{Future, Task}
//import com.thoughtworks.future.Future
//import com.thoughtworks.future.Future.Constant
//import com.thoughtworks.future.concurrent.Execution.JumpInto
//import com.thoughtworks.future.sde.task, task.AwaitOps
import shapeless.HList
import org.lwjgl.opencl.CL10._

import scala.annotation.tailrec
import scala.collection.mutable
import scala.concurrent.ExecutionContext
import com.thoughtworks.each.Monadic._

//
//final case class DifferentiableKernel[OutputData, OutputDelta](
//    context: OpenCL.Context,
//    commandQueue: CommandQueue,
//    @transient val forwardKernel: Future.Stateful[Kernel],
//    backwardExpressions: Map[Symbol, DslExpression]
//) extends Layer
//    with IsClosed {
//
//  override type Input = Map[Symbol, Tape]
//
//  override type Output = Tape.Aux[OutputData, OutputDelta]
//
//  @transient
//  private var forwardKernel: Future.Stateful[Kernel] = {
//
////    KernelDefinition("forward", Seq(), Seq())
//    // context.createProgramWithSource()
//  }
//
//  @transient
//  private val backwardKernels = mutable.Map.empty[Map[Symbol, Boolean], Kernel]
//
//  override def forward(input: Input): Stateless[Aux[OutputData, OutputDelta]] = ???
//
//  override protected def forceClose(): Unit = {
//    context.close()
//  }
//}

object DifferentiableKernel {

  trait InputMetadata {

    type Data
    type Delta

    def dataMemory: Memory[Data]

    def deltaMemory: Memory[Delta]

    def dataType: DslType

    def deltaType: DslType

  }

  object DifferentiableExpression {

    final case class FloatLiteral(data: Float) extends DifferentiableExpression {

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

      override def backward: (DslExpression) => Map[Any, DslExpression] = ???
    }

    private[DifferentiableExpression] final val OutputId = new AnyRef
    private[DifferentiableExpression] final val ForwardKernelName = "forward"
    private[DifferentiableExpression] type Setter[-OutputData] = (Kernel, Map[Any, Tape]) => Unit
    private[DifferentiableExpression] val EmptySetter: Setter[Any] = (kernel, input) => ()

  }

  trait DifferentiableExpression {
    import DifferentiableExpression._

    def inputMetadataMap: Map[Any, InputMetadata]

    type OutputElementData
    type OutputElementDelta

    def outputDataMemory: Memory[OutputElementData]

    def outputDeltaMemory: Memory[OutputElementDelta]

    def outputDataType: DslType

    def outputDeltaType: DslType

    def forward: DslExpression

    def backward: DslExpression => Map[Any, DslExpression]

    /**
      * @param executor on which the compilation runs
      */
    def compile(context: OpenCL.Context, device: Device, commandQueue: CommandQueue)(
        implicit executor: ExecutionContext): RAIITask[(Int, Map[Any, Tape]) => RAIITask[
      Tape.Aux[OpenCL.Buffer[OutputElementData], OpenCL.Buffer[OutputElementDelta]]]] = throwableMonadic[RAIITask] {

      RAIITask.jump().each

      val outputParameter = Parameter(OutputId, DslType.DslBuffer(outputDataType))

      val (parameters, inputSetter) = inputMetadataMap.view.zipWithIndex
        .foldRight[(List[Parameter], Setter[OutputElementData])]((outputParameter :: Nil, EmptySetter)) {
          case (((symbol, metadata), i), (accumulatedParameters, accumulatedSetter)) =>
            val parameter = Parameter(symbol, metadata.dataType)
            val setter: Setter[OutputElementData] = { (kernel, input) =>
              val inputData = input(symbol)
              kernel.setArg[inputData.type](i, inputData)(
                inputMetadataMap(symbol).dataMemory.asInstanceOf[Memory[inputData.type]])
              accumulatedSetter(kernel, input)
            }
            (parameter :: accumulatedParameters, setter)
        }

      def forwardKernelDefinition = {
        val outputIndex = {
          // TODO: Support n-dimension Array
          DslExpression.GetGlobalId(DslExpression.IntLiteral(0))
        }
        val effect = DslEffect.Update(DslExpression.Identifier(OutputId), outputIndex, forward, outputDataType)
        KernelDefinition(ForwardKernelName, parameters, Seq(effect))
      }

      val forwardSource = OpenCLCodeGenerator.generateSourceCode(forwardKernelDefinition).toArray[CharSequence]

      val program = RAIITask.managed(context.createProgramWithSource(forwardSource)).each
      RAIITask.unmanaged(program.build()).each
      (program, inputSetter)

      val backwardKernels = mutable.Map.empty[Map[Any, Boolean], Future[Kernel]]
//
//      new DifferentiableKernel {
//
//        // TODO: Change OutputData and OutputDelta to a pair of OpenCL.Buffer and OpenCL.Event
//        override type OutputData = OpenCL.Buffer[OutputElementData]
//        override type OutputDelta = OpenCL.Buffer[OutputElementDelta]
//
//        override def close(): Unit = ???
//
////        override def forward(input: Input) = monadic[Future] {
////          val (program, inputSetter) = forwardProgram.!
////          val kernel = program.createKernel(ForwardKernelName)
////          val (expectedSize, inputParameterMap) = input
////          val outputBuffer = context.createBuffer[OutputElementData](expectedSize)(outputDataMemory)
////          inputSetter(kernel, inputParameterMap)
////          kernel.setArg(inputMetadataMap.size, outputBuffer)
////          val event = commandQueue.enqueueNDRangeKernel(kernel, Seq(GlobalWorkSizeOnlyDimension(Address(expectedSize))))
////          try {
////            event.each
////          } finally {
////            event.close()
////          }
////          new Tapes.Fill(inputParameterMap, outputBuffer)
////          ???
////        }
//        override def forward(expectedOutputSize: Int, inputParameterMap: Map[Any, Tape])
//          : RAIITask[Tape.Aux[OpenCL.Buffer[OutputElementData], OpenCL.Buffer[OutputElementDelta]]] = {}
//      }

      { (expectedSize, inputParameterMap) =>
        throwableMonadic[RAIITask] {
          val kernel = program.createKernel(ForwardKernelName)
          val outputBuffer = context.createBuffer[OutputElementData](expectedSize)(outputDataMemory)
          inputSetter(kernel, inputParameterMap)
          kernel.setArg(inputMetadataMap.size, outputBuffer)
          val event =
            RAIITask
              .managed(
                commandQueue.enqueueNDRangeKernel(kernel, Seq(GlobalWorkSizeOnlyDimension(Address(expectedSize)))))
              .each
          event.waitForComplete()
          new Tape {
            override def data: OpenCL.Buffer[OutputElementData] = outputBuffer

            override def backward(delta: RAIITask[_ <: OpenCL.Buffer[OutputElementDelta]]): Future[Unit] = ???

            override type Data = OpenCL.Buffer[OutputElementData]
            override type Delta = OpenCL.Buffer[OutputElementDelta]
          }
        }
      }
    }

  }

}
