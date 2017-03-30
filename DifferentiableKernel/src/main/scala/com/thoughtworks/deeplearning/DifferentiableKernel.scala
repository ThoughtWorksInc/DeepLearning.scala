package com.thoughtworks.deeplearning

import java.util.concurrent.Executor

import com.thoughtworks.deeplearning.DifferentiableKernel._
import com.thoughtworks.deeplearning.Layer.Tape
import com.thoughtworks.deeplearning.Memory.Address
import com.thoughtworks.deeplearning.OpenCL.CommandQueue.GlobalWorkSizeOnlyDimension
import com.thoughtworks.deeplearning.OpenCL.{CommandQueue, Device, Kernel, Program}
import com.thoughtworks.deeplearning.OpenCLCodeGenerator.DslExpression.GetGlobalId
import com.thoughtworks.deeplearning.OpenCLCodeGenerator._
import com.thoughtworks.future.Future
import com.thoughtworks.future.Future.Constant
import com.thoughtworks.future.concurrent.Execution.JumpInto
import com.thoughtworks.future.sde.task,task.AwaitOps
import shapeless.HList
import org.lwjgl.opencl.CL10._

import scala.annotation.tailrec
import scala.collection.mutable
import scala.concurrent.ExecutionContext

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait DifferentiableKernel extends Layer with AutoCloseable { // TODO: rename to fill
  type OutputData
  type OutputDelta

  private type OutputSize = Int

  override type Input = (OutputSize, Map[Any, Tape])

  override type Output = Tape.Aux[OutputData, OutputDelta]

}

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

  type Aux[+OutputData0, -OutputDelta0] = DifferentiableKernel {
    type OutputData <: OutputData0
    type OutputDelta >: OutputDelta0
  }

  object Tapes {
//    final class Fill[OutputElementData, OutputElementDelta](inputParameterMap: Map[Any, Tape],
//                                                            override val value: OpenCL.Buffer[OutputElementData])
//        extends CumulativeTape {
//      override def isTrainable: Boolean = ???
//
//      private var deltaAccumulator: Future[Option[Delta]] = {
//        Future(None)
//      }
//
//      override def backward(delta: Delta) = ???
//
//      override type Delta = OpenCL.Buffer[OutputElementDelta]
//      override type Data = OpenCL.Buffer[OutputElementData]
//
//      private def flush(deltaFuture: Future.Stateful[Option[Delta]]): Future[Unit] = {
//        implicit def catcher = PartialFunction.empty
//        for (deltaOption <- deltaFuture) {
//          deltaOption match {
//            case None =>
//            case Some(delta) =>
//              ???
//          }
//        }
//      }
//
//      override protected def flush(): Future[Unit] = {
//        flush(synchronized {
//          val delta = deltaAccumulator
//          deltaAccumulator = new Constant(None)
//          delta
//        })
//      }
//
//      override protected def closeUpstreams(): Future[Unit] = Future {
//        for (upstream <- futureSeq(inputParameterMap.values)) {
//          upstream.close()
//        }
//      }
//    }
  }

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
    def compile(context: OpenCL.Context, device: Device, commandQueue: CommandQueue, executor: ExecutionContext)
      : DifferentiableKernel.Aux[OpenCL.Buffer[OutputElementData], OpenCL.Buffer[OutputElementDelta]] = {

      val forwardProgram = Future.completeWith(task {
        JumpInto(executor).!

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

        val program = context.createProgramWithSource(forwardSource)
        program.build().!
        (program, inputSetter)
      })

      val backwardKernels = mutable.Map.empty[Map[Any, Boolean], Future[Kernel]]

      new DifferentiableKernel {

        // TODO: Change OutputData and OutputDelta to a pair of OpenCL.Buffer and OpenCL.Event
        override type OutputData = OpenCL.Buffer[OutputElementData]
        override type OutputDelta = OpenCL.Buffer[OutputElementDelta]

        override def close(): Unit = ???

        override def forward(input: Input) = Future {
          val (program, inputSetter) = forwardProgram.!
          val kernel = program.createKernel(ForwardKernelName)
          val (expectedSize, inputParameterMap) = input
          val outputBuffer = context.createBuffer[OutputElementData](expectedSize)(outputDataMemory)
          inputSetter(kernel, inputParameterMap)
          kernel.setArg(inputMetadataMap.size, outputBuffer)
          val event =
            commandQueue.enqueueNDRangeKernel(kernel, Seq(GlobalWorkSizeOnlyDimension(Address(expectedSize))))
          try {
            event.!
          } finally {
            event.close()
          }
          new Tapes.Fill(inputParameterMap, outputBuffer)
        }
      }
    }

  }

}
