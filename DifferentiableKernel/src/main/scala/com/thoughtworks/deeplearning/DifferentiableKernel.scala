package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Memory.Address
import com.thoughtworks.deeplearning.OpenCL.CommandQueue.GlobalWorkSizeOnlyDimension
import com.thoughtworks.deeplearning.OpenCL.{CommandQueue, Device, Kernel}
import com.thoughtworks.deeplearning.OpenCLCodeGenerator.DslType.{DslBuffer, DslDouble, DslFloat, DslInt}
import com.thoughtworks.deeplearning.OpenCLCodeGenerator._
import com.thoughtworks.each.Monadic._
import com.thoughtworks.raii.RAIITask
import com.thoughtworks.raii.ResourceFactoryT.ResourceT
import shapeless._
import shapeless.labelled._

import scala.concurrent.ExecutionContext
import scala.language.higherKinds
import scala.util.control.NonFatal
import scalaz.concurrent.Future
import scalaz.{\/, \/-}

object DifferentiableKernel {

  final case class Weight() extends Tape{
    override def data: Data = ???

    override def backward[CovariantDelta <: Delta](outputDelta: RAIITask[CovariantDelta]): Future[Unit] = ???
  }

  final case class PendingBuffer[Element](buffer: OpenCL.Buffer[Element], events: List[OpenCL.Event])

  private[DifferentiableKernel] trait StaticDslTypeExtractor {
    type AbstractType[A] <: DslType

    implicit def dslDouble: AbstractType[Double]
    implicit def dslFloat: AbstractType[Float]
    implicit def dslInt: AbstractType[Int]
    implicit def dslBuffer[Element: AbstractType]: AbstractType[PendingBuffer[Element]]
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

  final case class OpenCLLayer[OutputElementData, OutputElementDelta, LocalDelta <: HList](
      data: StaticDslExpression[OutputElementData],
      jacobian: LocalDelta) {

    import OpenCLLayer._

    def compile(context: OpenCL.Context, commandQueue: CommandQueue, semaphore: AsynchronousSemaphore)(
        implicit compiler: Compiler[OutputElementData, OutputElementDelta, LocalDelta],
        outputDataMemory: Memory[OutputElementData],
        outputDeltaMemory: Memory[OutputElementDelta],
        outputDataType: StaticDslType[OutputElementData],
        outputDeltaType: StaticDslType[OutputElementDelta],
        executor: ExecutionContext): RAIITask[(Int, compiler.ParameterRecord) => RAIITask[
      Tape.Aux[PendingBuffer[OutputElementData], PendingBuffer[OutputElementDelta]]]] = throwableMonadic[RAIITask] {

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
          val outputBuffer = context.createBuffer[OutputElementData](expectedSize)(outputDataMemory)

          compiler.setKernelInputArguments(kernel, 1, inputParameterMap)
          kernel.setArg(0, outputBuffer)

          RAIITask.unmanaged(semaphore.acquire()).each
          val event = try {
            RAIITask
              .managed(
                commandQueue.enqueueNDRangeKernel(kernel, Seq(GlobalWorkSizeOnlyDimension(Address(expectedSize)))))
              .each
          } catch {
            case e if NonFatal(e) =>
              semaphore.release().run
              (throw e): OpenCL.Event
          }
          event.waitForComplete().unsafePerformAsync { _ =>
            semaphore.release().run
          }

          RAIITask.unmanaged(event.waitForComplete()).each
          val pendingBuffer = PendingBuffer(outputBuffer, List(event))
          new Tape {

            override def data: Data = pendingBuffer

            override def backward[OutputDeltaBuffer <: PendingBuffer[OutputElementDelta]](
                outputDeltaTask: RAIITask[OutputDeltaBuffer]): Future[Unit] = {
              Future.suspend {
                Future.now(()) // TODO: backward
              }

            }

            // TODO: Change OutputData and OutputDelta to a pair of OpenCL.Buffer and OpenCL.Event
            override type Data = PendingBuffer[OutputElementData]
            override type Delta = PendingBuffer[OutputElementDelta]
          }: Tape.Aux[PendingBuffer[OutputElementData], PendingBuffer[OutputElementDelta]]
        }
      }
    }
  }

  trait GetElement[Operand0, Operand1] extends DepFn2[Operand0, Operand1]
  object GetElement {
    type Aux[Operand0, Operand1, Out0] = GetElement[Operand0, Operand1] {
      type Out = Out0
    }

    implicit def bufferJacobianGetElement[Data, Delta](implicit one: One.Aux[StaticDslExpression[Delta]])
      : GetElement.Aux[JacobianMatrix[Data, Delta], StaticDslExpression[Int], JacobianMatrix.Row[Data, Delta]] = {
      new GetElement[JacobianMatrix[Data, Delta], StaticDslExpression[Int]] {
        type Out = JacobianMatrix.Row[Data, Delta]
        override def apply(bufferJacobian: JacobianMatrix[Data, Delta], index: StaticDslExpression[Int]): Out = {
          bufferJacobian match {
            case JacobianMatrix.Identity() =>
              JacobianMatrix.Row.Sparse(index, one())

          }
        }
      }
    }

    implicit def hconsGetElement[Key, Head, Tail <: HList, Operand1, HeadOut, TailOut <: HList](
        implicit headGetElement: GetElement.Aux[Head, Operand1, HeadOut],
        tailGetElement: GetElement.Aux[Tail, Operand1, TailOut])
      : GetElement.Aux[FieldType[Key, Head] :: Tail, Operand1, FieldType[Key, HeadOut] :: TailOut] =
      new GetElement[FieldType[Key, Head] :: Tail, Operand1] {
        override type Out = FieldType[Key, HeadOut] :: TailOut

        override def apply(operand0: FieldType[Key, Head] :: Tail, operand1: Operand1): Out = {
          field[Key](headGetElement(operand0.head, operand1)) :: tailGetElement(operand0.tail, operand1)
        }
      }

    implicit def hnilGetElement[Operand1]: GetElement.Aux[HNil, Operand1, HNil] = new GetElement[HNil, Operand1] {
      override type Out = HNil

      override def apply(operand0: HNil, operand1: Operand1): HNil.type = HNil
    }
  }
  trait Plus[Operand0, Operand1] extends DepFn2[Operand0, Operand1]
  object Plus {
    type Aux[Operand0, Operand1, Out0] = Plus[Operand0, Operand1] {
      type Out = Out0
    }
    implicit def floatPlus = new Plus[StaticDslExpression[Float], StaticDslExpression[Float]] {
      override type Out = StaticDslExpression[Float]
      override def apply(operand0: StaticDslExpression[Float], operand1: StaticDslExpression[Float]): Out = {
        StaticDslExpression[Float](DslExpression.Plus(operand0, operand1, DslType.DslFloat))
      }
    }

    implicit def pickFieldExistsInOperand0Only[K, V, T <: HList, M <: HList, Out0 <: HList](
        implicit mt: Plus.Aux[T, M, Out0],
        lacksKey: shapeless.ops.record.LacksKey[M, K]): Aux[FieldType[K, V] :: T, M, FieldType[K, V] :: Out0] =
      new Plus[FieldType[K, V] :: T, M] {
        type Out = FieldType[K, V] :: Out0
        def apply(l: FieldType[K, V] :: T, m: M): Out = l.head :: mt(l.tail, m)
      }
    implicit def mergeFieldExistsBoth[K, V0, V1, V, T <: HList, M <: HList, MT <: HList, Out0 <: HList](
        implicit rm: shapeless.ops.record.Remover.Aux[M, K, (V1, MT)],
        mt: Plus.Aux[T, MT, Out0],
        callback: Plus.Aux[V0, V1, V]): Aux[FieldType[K, V0] :: T, M, FieldType[K, V] :: Out0] = {
      new Plus[FieldType[K, V0] :: T, M] {
        type Out = FieldType[K, V] :: Out0
        def apply(l: FieldType[K, V0] :: T, m: M): Out = {
          val (mv, mr) = rm(m)
          val up = field[K](callback(l.head: V0, mv))
          up :: mt(l.tail, mr)
        }
      }
    }

    implicit def mergeNil[M <: HList]: Aux[HNil, M, M] = {
      new Plus[HNil, M] {
        type Out = M
        def apply(l: HNil, m: M): Out = m
      }
    }
  }
  trait Times[Operand0, Operand1] extends DepFn2[Operand0, Operand1]
  abstract class LowPriorityTimes0 {
    implicit def swap[Operand0, Operand1, Out0](
        implicit times: Times.Aux[Operand1, Operand0, Out0]): Times.Aux[Operand0, Operand1, Out0] =
      new Times[Operand0, Operand1] {
        override type Out = Out0

        override def apply(operand0: Operand0, operand1: Operand1): Out = times(operand1, operand0)
      }
  }
  object Times extends LowPriorityTimes0 {
    type Aux[Operand0, Operand1, Out0] = Times[Operand0, Operand1] {
      type Out = Out0
    }

    implicit def floatTimes = new Times[StaticDslExpression[Float], StaticDslExpression[Float]] {
      override type Out = StaticDslExpression[Float]
      override def apply(operand0: StaticDslExpression[Float], operand1: StaticDslExpression[Float]): Out = {
        StaticDslExpression[Float](DslExpression.Times(operand0, operand1, DslType.DslFloat))
      }
    }

    implicit def hconsTimes[Key, Head, Tail <: HList, Operand1, HeadOut, TailOut <: HList](
        implicit headTimes: Times.Aux[Head, Operand1, HeadOut],
        tailTimes: Times.Aux[Tail, Operand1, TailOut])
      : Times.Aux[FieldType[Key, Head] :: Tail, Operand1, FieldType[Key, HeadOut] :: TailOut] =
      new Times[FieldType[Key, Head] :: Tail, Operand1] {
        override type Out = FieldType[Key, HeadOut] :: TailOut

        override def apply(operand0: FieldType[Key, Head] :: Tail, operand1: Operand1): Out = {
          field[Key](headTimes(operand0.head, operand1)) :: tailTimes(operand0.tail, operand1)
        }
      }

    implicit def hnilTimes[Operand1]: Times.Aux[HNil, Operand1, HNil] = new Times[HNil, Operand1] {
      override type Out = HNil

      override def apply(operand0: HNil, operand1: Operand1): HNil.type = HNil
    }
  }
  trait Zero extends DepFn0
  object Zero {
    type Aux[Out0] = Zero {
      type Out = Out0
    }

    implicit object FloatZero extends Zero {
      type Out = StaticDslExpression[Float]
      override def apply(): Out = StaticDslExpression(DslExpression.FloatLiteral(0.0f))
    }

    implicit object HNilZero extends Zero {
      type Out = HNil
      override def apply(): HNil = HNil
    }
  }
  trait One extends DepFn0
  object One {
    type Aux[Out0] = One {
      type Out = Out0
    }

    implicit object FloatOne extends One {
      override def apply(): Out = StaticDslExpression(DslExpression.FloatLiteral(1.0f))

      override type Out = StaticDslExpression[Float]
    }
  }

  object OpenCLLayer {
    private[deeplearning] final val OutputId = new AnyRef
    @inline private[deeplearning] final val ForwardKernelName = "forward"
    @inline private[deeplearning] final def backwardKernelName(index: Int) = raw"""backward_$index"""

    def floatLiteral(data: Float): OpenCLLayer[Float, Float, HNil] = {
      OpenCLLayer[Float, Float, HNil](StaticDslExpression[Float](DslExpression.FloatLiteral(data)), HNil)
    }
    def intLiteral(data: Int): OpenCLLayer[Int, Float, HNil] = {
      OpenCLLayer[Int, Float, HNil](StaticDslExpression[Int](DslExpression.IntLiteral(data)), HNil)
    }

    def bufferIdentifier[Data, Delta](
        key: Witness): OpenCLLayer[PendingBuffer[Data],
                                   PendingBuffer[Delta],
                                   FieldType[key.T, JacobianMatrix[Data, Delta]] :: HNil] = {
      OpenCLLayer[PendingBuffer[Data], PendingBuffer[Delta], FieldType[key.T, JacobianMatrix[Data, Delta]] :: HNil](
        StaticDslExpression(DslExpression.Identifier(key.value)),
        field[key.T](JacobianMatrix.Identity[Data, Delta]()) :: HNil
      )
    }

    def getGlobalId[LocalDelta <: HList](dimension: OpenCLLayer[Int, Float, LocalDelta])(
        implicit zero: Zero.Aux[LocalDelta]): OpenCLLayer[Int, Float, LocalDelta] = {
      OpenCLLayer[Int, Float, LocalDelta](
        StaticDslExpression(DslExpression.GetGlobalId(dimension.data)),
        zero()
      )
    }

    def getElement[ElementData,
                   ElementDelta,
                   BufferLocalDelta <: HList,
                   IndexLocalDelta <: HList,
                   ElementLocalDelta <: HList,
                   LocalDelta <: HList](
        buffer: OpenCLLayer[PendingBuffer[ElementData], PendingBuffer[ElementDelta], BufferLocalDelta],
        index: OpenCLLayer[Int, Float, IndexLocalDelta])(
        implicit elementDataType: StaticDslType[ElementData],
        zero: Zero.Aux[IndexLocalDelta],
        getDeltaElement: GetElement.Aux[BufferLocalDelta, StaticDslExpression[Int], ElementLocalDelta],
        plus: Plus.Aux[ElementLocalDelta, IndexLocalDelta, LocalDelta]
    ): OpenCLLayer[ElementData, ElementDelta, LocalDelta] = {
      OpenCLLayer[ElementData, ElementDelta, LocalDelta](
        StaticDslExpression[ElementData](DslExpression.GetElement(buffer.data, index.data, elementDataType)),
        plus(getDeltaElement(buffer.jacobian, index.data), zero())
      )
    }

  }

  import OpenCLLayer._

  object JacobianMatrix {
    final case class Identity[Data, Delta]() extends JacobianMatrix[Data, Delta]

    /** Partial derivatives of a scalar-valued function */
    sealed trait Row[Data, Delta]

    object Row {
      final case class Sparse[Data, Delta](index: StaticDslExpression[Int], value: StaticDslExpression[Delta])
          extends Row[Data, Delta]
    }

    /** A partial derivative of a vector-valued function */
    sealed trait Column[Data, Delta]
  }

  /** Partial derivatives of a vector-valued function */
  sealed trait JacobianMatrix[Data, Delta]

  trait InputCompiler[Key, LocalDelta] {
    type Input <: Tape

    def forwardParameter: Parameter

    def setArgument(kernel: Kernel, index: Int, input: Input): Unit

    def borrowEvents(input: Input): List[OpenCL.Event]
  }

  object InputCompiler {

    type Aux[Key, LocalDelta, Input0] = InputCompiler[Key, LocalDelta] {
      type Input = Input0
    }

    implicit def bufferInputCompile[Key, InputElementData, InputElementDelta](
        implicit witness: Witness.Aux[Key],
        elementDataType: StaticDslType[InputElementData])
      : InputCompiler.Aux[Key,
                          JacobianMatrix.Row[InputElementData, InputElementDelta],
                          Tape.Aux[PendingBuffer[InputElementData], PendingBuffer[InputElementDelta]]] =
      new InputCompiler[Key, JacobianMatrix.Row[InputElementData, InputElementDelta]] {

        override type Input = Tape.Aux[PendingBuffer[InputElementData], PendingBuffer[InputElementDelta]]
        override def forwardParameter: Parameter = Parameter(witness.value, DslType.DslBuffer(elementDataType))

        override def setArgument(kernel: Kernel, index: Int, input: Input): Unit = {
          kernel.setArg[OpenCL.Buffer[InputElementData]](index, input.data.buffer)
        }

        override def borrowEvents(
            input: Tape.Aux[PendingBuffer[InputElementData], PendingBuffer[InputElementDelta]]): List[OpenCL.Event] = {
          input.data.events
        }
      }

    def apply[Key, InputElementData](implicit inputCompiler: InputCompiler[Key, InputElementData])
      : InputCompiler.Aux[Key, InputElementData, inputCompiler.Input] = {
      inputCompiler
    }

  }

  trait Compiler[OutputElementData, OutputElementDelta, LocalDelta <: HList] {
    type ParameterRecord <: HList

    def forwardInputParameters: List[Parameter]

    def forwardParameters(implicit outputDataType: StaticDslType[OutputElementData]): List[Parameter] =
      Parameter(OutputId, DslType.DslBuffer(outputDataType)) :: forwardInputParameters

    def setKernelInputArguments(kernel: Kernel, startIndex: Int, parameters: ParameterRecord)

    def borrowEvents(parameters: ParameterRecord): List[OpenCL.Event]
  }

  object Compiler {
    type Aux[OutputElementData, OutputElementDelta, LocalDelta <: HList, ParameterRecord0] =
      Compiler[OutputElementData, OutputElementDelta, LocalDelta] {
        type ParameterRecord = ParameterRecord0
      }

    def apply[OutputElementData, OutputElementDelta, LocalDelta <: HList](
        implicit compiler: Compiler[OutputElementData, OutputElementDelta, LocalDelta])
      : Compiler.Aux[OutputElementData, OutputElementDelta, LocalDelta, compiler.ParameterRecord] = compiler

    implicit def hnilFill[OutputElementData, OutputElementDelta]
      : Compiler.Aux[OutputElementData, OutputElementDelta, HNil, HNil] =
      new Compiler[OutputElementData, OutputElementDelta, HNil] {
        override type ParameterRecord = HNil

        override def forwardInputParameters: Nil.type = Nil

        override def setKernelInputArguments(kernel: Kernel, startIndex: Int, parameters: HNil): Unit = {}

        override def borrowEvents(parameters: HNil): List[OpenCL.Event] = Nil
      }

    implicit def hconsFill[OutputElementData,
                           OutputElementDelta0,
                           Key,
                           LocalDeltaHead,
                           LocalDeltaTail <: HList,
                           Input,
                           TailParameterRecord <: HList](
        implicit headInputCompiler: InputCompiler.Aux[Key, LocalDeltaHead, Input],
        tailCompiler: Compiler.Aux[OutputElementData, OutputElementDelta0, LocalDeltaTail, TailParameterRecord])
      : Compiler.Aux[OutputElementData,
                     OutputElementDelta0,
                     FieldType[Key, LocalDeltaHead] :: LocalDeltaTail,
                     FieldType[Key, Input] :: TailParameterRecord] =
      new Compiler[OutputElementData, OutputElementDelta0, FieldType[Key, LocalDeltaHead] :: LocalDeltaTail] {
        override type ParameterRecord = FieldType[Key, Input] :: TailParameterRecord
        override def forwardInputParameters: List[Parameter] =
          headInputCompiler.forwardParameter :: tailCompiler.forwardInputParameters

        override def setKernelInputArguments(kernel: Kernel,
                                             startIndex: Int,
                                             parameters: FieldType[Key, Input] :: TailParameterRecord): Unit = {
          headInputCompiler.setArgument(kernel, startIndex, parameters.head)
          tailCompiler.setKernelInputArguments(kernel, startIndex + 1, parameters.tail)
        }

        override def borrowEvents(parameters: ::[FieldType[Key, Input], TailParameterRecord]): List[OpenCL.Event] = {
          headInputCompiler.borrowEvents(parameters.head) ::: tailCompiler.borrowEvents(parameters.tail)
        }
      }

  }

}
