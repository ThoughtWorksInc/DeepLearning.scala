package com.thoughtworks.deeplearning

import java.util.concurrent.Executor

import com.qifun.statelessFuture.Future
import com.qifun.statelessFuture.Future.Stateless
import com.qifun.statelessFuture.util._
import com.thoughtworks.deeplearning.DifferentiableKernel._
import com.thoughtworks.deeplearning.Layer.Tape
import com.thoughtworks.deeplearning.OpenCL.CommandQueue.Dimension
import com.thoughtworks.deeplearning.OpenCL.{CommandQueue, Device, Kernel, Program}
import com.thoughtworks.deeplearning.OpenCLCodeGenerator.DslExpression.GetGlobalId
import com.thoughtworks.deeplearning.OpenCLCodeGenerator._
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

  type OutputBufferSize = Tape.Aux[Int, Nothing]
  override type Input = (OutputBufferSize, Map[Any, Tape])

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
    final class Fill[OutputElementData, OutputElementDelta](inputParameterMap: Map[Any, Tape],
                                                            override val value: OpenCL.Buffer[OutputElementData])
        extends CumulativeTape {
      override def isTrainable: Boolean = ???

      private var deltaAccumulator: Future.Stateful[Option[Delta]] = {
        new Constant(None)
      }

      override def forceBackward(delta: Delta) = ???

      override type Delta = OpenCL.Buffer[OutputElementDelta]
      override type Data = OpenCL.Buffer[OutputElementData]

      private def rawBackward(deltaFuture: Future.Stateful[Option[Delta]]): Unit = {
        implicit def catcher = PartialFunction.empty
        for (deltaOption <- deltaFuture) {
          deltaOption match {
            case None =>
            case Some(delta) =>
              ???
          }
        }
      }

      override protected def flush(): Unit = {
        rawBackward(synchronized {
          val delta = deltaAccumulator
          deltaAccumulator = new Constant(None)
          delta
        })
      }

      override protected def closeUpstreams(): Unit = {
        for (upstream <- inputParameterMap.values) {
          upstream.close()
        }
      }
    }
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

      val forwardProgram = Promise[(Program, Setter[OutputElementData])]
      forwardProgram.completeWith(Future {
        JumpInto(executor).await

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
        program.build().await
        (program, inputSetter)
      })

      val backwardKernels = mutable.Map.empty[Map[Any, Boolean], Future.Stateful[Kernel]]

      new DifferentiableKernel {

        // TODO: Change OutputData and OutputDelta to a pair of OpenCL.Buffer and OpenCL.Event
        override type OutputData = OpenCL.Buffer[OutputElementData]
        override type OutputDelta = OpenCL.Buffer[OutputElementDelta]

        override def close(): Unit = ???

        override def forward(input: Input) = Future {
          val (program, inputSetter) = forwardProgram.await
          val kernel = program.createKernel(ForwardKernelName)
          val (size, inputParameterMap) = input
          val outputBuffer = context.createBuffer[OutputElementData](size.value)(outputDataMemory)
          inputSetter(kernel, inputParameterMap)
          kernel.setArg(inputMetadataMap.size, outputBuffer)

          val dimension = if (device.capabilities.OpenCL20) {
            Dimension(0L, size.value, device.maxWorkItemSizes.get(0))
          } else {
            val expectedSize = size.value
            val localWorkSize = math.min(device.maxWorkGroupSize.toLong, device.maxWorkItemSizes.get(0))
            val mod = expectedSize % localWorkSize
            if (mod == 0) {
              Dimension(0L, expectedSize, localWorkSize)
            } else {
              Dimension(0L, expectedSize + localWorkSize - mod, localWorkSize)
            }
          }

          val event = commandQueue.ndRangeKernel(kernel, Seq(dimension))
          try {
            event.await
          } finally {
            event.close()
          }
          new Tapes.Fill(inputParameterMap, outputBuffer)
        }
      }
    }

  }

}
