package com.thoughtworks.deeplearning

import java.io.Closeable
import java.nio.{ByteBuffer, IntBuffer}

import org.lwjgl.opencl._
import CL10._
import CL11._
import CL20._
import com.thoughtworks.deeplearning.Closeables.{AssertionAutoCloseable, AssertionFinalizer}
import org.lwjgl.{BufferUtils, PointerBuffer}
import org.lwjgl.system.MemoryUtil.{memASCII, _}
import org.lwjgl.system.MemoryStack._
import org.lwjgl.system.Pointer._

import scala.collection.mutable
import com.thoughtworks.deeplearning.Memory.{Address, Box}
import org.lwjgl.system.{JNI, MemoryStack, MemoryUtil, Pointer}

import scala.util.control.Exception.Catcher
import scala.util.control.TailCalls
import scala.util.control.TailCalls.TailRec
import scala.util.{Failure, Success, Try}
import scalaz.{-\/, \/, \/-}
import scalaz.concurrent.{Future, Task}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object OpenCL {

  object Exceptions {

    final class DeviceNotFound extends IllegalArgumentException

    final class DeviceNotAvailable extends IllegalStateException
    final class CompilerNotAvailable extends IllegalStateException
    final class MemObjectAllocationFailure extends IllegalStateException
    final class OutOfResources extends IllegalStateException
    final class OutOfHostMemory extends IllegalStateException
    final class ProfilingInfoNotAvailable extends IllegalStateException
    final class MemCopyOverlap extends IllegalStateException
    final class ImageFormatMismatch extends IllegalStateException
    final class ImageFormatNotSupported extends IllegalStateException
    final class BuildProgramFailure extends IllegalStateException
    final class MapFailure extends IllegalStateException

    final class InvalidValue extends IllegalArgumentException
    final class InvalidDeviceType extends IllegalArgumentException
    final class InvalidPlatform extends IllegalArgumentException
    final class InvalidDevice extends IllegalArgumentException
    final class InvalidContext extends IllegalArgumentException
    final class InvalidQueueProperties extends IllegalArgumentException
    final class InvalidCommandQueue extends IllegalArgumentException
    final class InvalidHostPtr extends IllegalArgumentException
    final class InvalidMemObject extends IllegalArgumentException
    final class InvalidImageFormatDescriptor extends IllegalArgumentException
    final class InvalidImageSize extends IllegalArgumentException
    final class InvalidSampler extends IllegalArgumentException
    final class InvalidBinary extends IllegalArgumentException
    final class InvalidBuildOptions extends IllegalArgumentException
    final class InvalidProgram extends IllegalArgumentException
    final class InvalidProgramExecutable extends IllegalArgumentException
    final class InvalidKernelName extends IllegalArgumentException
    final class InvalidKernelDefinition extends IllegalArgumentException
    final class InvalidKernel extends IllegalArgumentException
    final class InvalidArgIndex extends IllegalArgumentException
    final class InvalidArgValue extends IllegalArgumentException
    final class InvalidArgSize extends IllegalArgumentException
    final class InvalidKernelArgs extends IllegalArgumentException
    final class InvalidWorkDimension extends IllegalArgumentException
    final class InvalidWorkGroupSize extends IllegalArgumentException
    final class InvalidWorkItemSize extends IllegalArgumentException
    final class InvalidGlobalOffset extends IllegalArgumentException
    final class InvalidEventWaitList extends IllegalArgumentException
    final class InvalidEvent extends IllegalArgumentException
    final class InvalidOperation extends IllegalArgumentException
    final class InvalidBufferSize extends IllegalArgumentException
    final class InvalidGlobalWorkSize extends IllegalArgumentException

    final class UnknownErrorCode(errorCode: Int) extends IllegalStateException

    def fromErrorCode(errorCode: Int): Exception = errorCode match {
      case CL_DEVICE_NOT_FOUND => new Exceptions.DeviceNotFound
      case CL_DEVICE_NOT_AVAILABLE => new Exceptions.DeviceNotAvailable
      case CL_COMPILER_NOT_AVAILABLE => new Exceptions.CompilerNotAvailable
      case CL_MEM_OBJECT_ALLOCATION_FAILURE => new Exceptions.MemObjectAllocationFailure
      case CL_OUT_OF_RESOURCES => new Exceptions.OutOfResources
      case CL_OUT_OF_HOST_MEMORY => new Exceptions.OutOfHostMemory
      case CL_PROFILING_INFO_NOT_AVAILABLE => new Exceptions.ProfilingInfoNotAvailable
      case CL_MEM_COPY_OVERLAP => new Exceptions.MemCopyOverlap
      case CL_IMAGE_FORMAT_MISMATCH => new Exceptions.ImageFormatMismatch
      case CL_IMAGE_FORMAT_NOT_SUPPORTED => new Exceptions.ImageFormatNotSupported
      case CL_BUILD_PROGRAM_FAILURE => new Exceptions.BuildProgramFailure
      case CL_MAP_FAILURE => new Exceptions.MapFailure
      case CL_INVALID_VALUE => new Exceptions.InvalidValue
      case CL_INVALID_DEVICE_TYPE => new Exceptions.InvalidDeviceType
      case CL_INVALID_PLATFORM => new Exceptions.InvalidPlatform
      case CL_INVALID_DEVICE => new Exceptions.InvalidDevice
      case CL_INVALID_CONTEXT => new Exceptions.InvalidContext
      case CL_INVALID_QUEUE_PROPERTIES => new Exceptions.InvalidQueueProperties
      case CL_INVALID_COMMAND_QUEUE => new Exceptions.InvalidCommandQueue
      case CL_INVALID_HOST_PTR => new Exceptions.InvalidHostPtr
      case CL_INVALID_MEM_OBJECT => new Exceptions.InvalidMemObject
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR => new Exceptions.InvalidImageFormatDescriptor
      case CL_INVALID_IMAGE_SIZE => new Exceptions.InvalidImageSize
      case CL_INVALID_SAMPLER => new Exceptions.InvalidSampler
      case CL_INVALID_BINARY => new Exceptions.InvalidBinary
      case CL_INVALID_BUILD_OPTIONS => new Exceptions.InvalidBuildOptions
      case CL_INVALID_PROGRAM => new Exceptions.InvalidProgram
      case CL_INVALID_PROGRAM_EXECUTABLE => new Exceptions.InvalidProgramExecutable
      case CL_INVALID_KERNEL_NAME => new Exceptions.InvalidKernelName
      case CL_INVALID_KERNEL_DEFINITION => new Exceptions.InvalidKernelDefinition
      case CL_INVALID_KERNEL => new Exceptions.InvalidKernel
      case CL_INVALID_ARG_INDEX => new Exceptions.InvalidArgIndex
      case CL_INVALID_ARG_VALUE => new Exceptions.InvalidArgValue
      case CL_INVALID_ARG_SIZE => new Exceptions.InvalidArgSize
      case CL_INVALID_KERNEL_ARGS => new Exceptions.InvalidKernelArgs
      case CL_INVALID_WORK_DIMENSION => new Exceptions.InvalidWorkDimension
      case CL_INVALID_WORK_GROUP_SIZE => new Exceptions.InvalidWorkGroupSize
      case CL_INVALID_WORK_ITEM_SIZE => new Exceptions.InvalidWorkItemSize
      case CL_INVALID_GLOBAL_OFFSET => new Exceptions.InvalidGlobalOffset
      case CL_INVALID_EVENT_WAIT_LIST => new Exceptions.InvalidEventWaitList
      case CL_INVALID_EVENT => new Exceptions.InvalidEvent
      case CL_INVALID_OPERATION => new Exceptions.InvalidOperation
      case CL_INVALID_BUFFER_SIZE => new Exceptions.InvalidBufferSize
      case CL_INVALID_GLOBAL_WORK_SIZE => new Exceptions.InvalidGlobalWorkSize
      case _ => new Exceptions.UnknownErrorCode(errorCode)
    }

  }

  def checkErrorCode(errorCode: Int): Unit = {
    errorCode match {
      case CL_SUCCESS =>
      case _ => throw Exceptions.fromErrorCode(errorCode)
    }
  }

  def platforms: Seq[Platform] = {
    val Array(numberOfPlatformIDs) = {
      val a = Array(0)
      checkErrorCode(clGetPlatformIDs(null, a))
      a
    }
    val stack = stackPush()
    try {
      val platformIdBuffer = stack.mallocPointer(numberOfPlatformIDs)
      checkErrorCode(clGetPlatformIDs(platformIdBuffer, null: IntBuffer))
      for (i <- (0 until platformIdBuffer.capacity).view) yield {
        val platformId = platformIdBuffer.get(i)
        val platformCapabilities = CL.createPlatformCapabilities(platformId)
        Platform(platformId, platformCapabilities)
      }
    } finally {
      stack.close()
    }
  }

  final case class Device private[OpenCL] (id: Long, capabilities: CLCapabilities) {

    def maxWorkItemDimensions: Int = intInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)

    def maxWorkItemSizes: PointerBuffer = {
      val size = maxWorkItemDimensions
      val buffer = BufferUtils.createPointerBuffer(size)
      checkErrorCode(clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_SIZES, buffer, null))
      buffer
    }

    def maxWorkGroupSize: Address = {
      addressInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE)
    }

    def intInfo(paramName: Int): Int = {
      val buffer = Array[Int](0)
      checkErrorCode(clGetDeviceInfo(id, paramName, buffer, null))
      val Array(value) = buffer
      value
    }

    def addressInfo(paramName: Int): Address = {
      val stack = stackPush()
      try {
        val buffer = stack.mallocPointer(1)
        checkErrorCode(clGetDeviceInfo(id, paramName, buffer, null))
        Address(buffer.get(0))
      } finally {
        stack.close()
      }
    }

    def longInfo(paramName: Int): Long = {
      val buffer = Array[Long](0L)
      checkErrorCode(clGetDeviceInfo(id, paramName, buffer, null))
      val Array(value) = buffer
      value
    }

    /**
      * Describes the command-queue properties supported by the device.
      * @see [[org.lwjgl.opencl.CL10.CL_DEVICE_QUEUE_PROPERTIES]]
      * @return
      */
    def queueProperties: Long = longInfo(CL_DEVICE_QUEUE_PROPERTIES)

    def deviceType: Long = longInfo(CL_DEVICE_TYPE)
  }

  object Context {

    override protected def finalize(): Unit = {
      callback.close()
    }

    @volatile
    var defaultLogger: (String, ByteBuffer) => Unit = { (errorInfo: String, data: ByteBuffer) =>
      // TODO: Add a test for in the case that Context is closed
      Console.err.println(raw"""An OpenCL notify comes out after its corresponding handler is freed
message: $errorInfo
data: $data""")
    }

    val callback: CLContextCallback = CLContextCallback.create(new CLContextCallbackI {
      override def invoke(errInfo: Long, privateInfo: Long, size: Long, userData: Long): Unit = {
        val errorInfo = memASCII(errInfo)
        val data = memByteBuffer(privateInfo, size.toInt)
        memGlobalRefToObject[(String, ByteBuffer) => Unit](userData) match {
          case null =>
            defaultLogger(memASCII(errInfo), memByteBuffer(privateInfo, size.toInt))
          case logger =>
            if (size.isValidInt) {
              logger(memASCII(errInfo), memByteBuffer(privateInfo, size.toInt))
            } else {
              throw new IllegalArgumentException(s"numberOfBytes($size) is too large")
            }
        }
      }
    })

  }

  final case class Platform private[OpenCL] (id: Long, capabilities: CLCapabilities) {

    def cpus: Seq[Device] = devicesByType(CL_DEVICE_TYPE_CPU)

    def gpus: Seq[Device] = devicesByType(CL_DEVICE_TYPE_GPU)

    def accelerators: Seq[Device] = devicesByType(CL_DEVICE_TYPE_ACCELERATOR)

    def devices: Seq[Device] = devicesByType(CL_DEVICE_TYPE_ALL)

    def devicesByType(deviceType: Int): Seq[Device] = {
      val Array(numberOfDevices) = {
        val a = Array(0)
        checkErrorCode(clGetDeviceIDs(Platform.this.id, deviceType, null, a))
        a
      }
      val stack = stackPush()
      try {
        val deviceIds = stack.mallocPointer(numberOfDevices)
        checkErrorCode(clGetDeviceIDs(Platform.this.id, deviceType, deviceIds, null: IntBuffer))
        for (i <- 0 until deviceIds.capacity()) yield {
          val deviceId = deviceIds.get(i)
          val deviceCapabilities = CL.createDeviceCapabilities(deviceId, Platform.this.capabilities)
          Device(deviceId, deviceCapabilities)
        }
      } finally {
        stack.close()
      }
    }

    def createContext(logger: (String, ByteBuffer) => Unit, forDevices: Device*): Context = {
      val stack = stackPush()
      try {
        val errorCodeBuffer = stack.ints(0)
        val contextProperties = stack.pointers(CL_CONTEXT_PLATFORM, Platform.this.id, 0)
        val deviceIds = stack.pointers(devices.map(_.id): _*)
        val context =
          clCreateContext(contextProperties, deviceIds, Context.callback, memNewWeakGlobalRef(logger), errorCodeBuffer)
        checkErrorCode(errorCodeBuffer.get(0))
        new Context(Address(context), logger)
      } finally {
        stack.close()
      }
    }

  }

  /**
    * @param logger keep the reference to keep the weak reference to logger
    */
  final class Context private[OpenCL] (val handle: Address, logger: AnyRef)
      extends AssertionAutoCloseable
      with AssertionFinalizer {

    def createProgramWithSource(sourceCode: TraversableOnce[CharSequence]): Program = {
      val stack = stackPush()
      try {
        val errorCodeBuffer = stack.ints(0)
        val programHandle = clCreateProgramWithSource(handle.toLong, sourceCode.toArray, errorCodeBuffer)
        checkErrorCode(errorCodeBuffer.get(0))
        new Program(Address(programHandle))
      } finally {
        stack.close()
      }
    }

    def createCommandQueue(device: Device, properties: Map[Int, Long]): CommandQueue = {
      if (device.capabilities.OpenCL20) {
        val cl20Properties = ((properties.view.flatMap { case (key, value) => Seq(key, value) }) ++ Seq(0L)).toArray
        val a = Array(0)
        val commandQueue = clCreateCommandQueueWithProperties(handle.toLong, device.id, cl20Properties, a)
        checkErrorCode(a(0))
        new CommandQueue(Address(commandQueue))
      } else {
        val cl10Properties = properties.getOrElse(CL_QUEUE_PROPERTIES, 0L)
        val a = Array(0)
        val commandQueue = clCreateCommandQueue(handle.toLong, device.id, cl10Properties, a)
        checkErrorCode(a(0))
        new CommandQueue(Address(commandQueue))
      }

    }

    def duplicate(): Context = {
      checkErrorCode(clRetainContext(handle.toLong))
      new Context(handle, logger)
    }

    override protected def forceClose(): Unit = {
      checkErrorCode(clReleaseContext(handle.toLong))
    }

    def createBuffer[Element](size: Long)(implicit sizeOf: Memory[Element]): Buffer[Element] = {
      val stack = stackPush()
      try {
        val errorCodeBuffer = stack.ints(0)
        val buffer =
          clCreateBuffer(handle.toLong, CL_MEM_READ_WRITE, sizeOf.numberOfBytesPerElement * size, errorCodeBuffer)
        checkErrorCode(errorCodeBuffer.get(0))
        new Buffer(Address(buffer))
      } finally {
        stack.pop()
      }
    }

  }
  object CommandQueue {

    final case class NDimensionBuffer(globalWorkOffset: PointerBuffer,
                                      globalWorkSize: PointerBuffer,
                                      localWorkSize: PointerBuffer)

    trait NDimensionBufferAllocator {
      def numberOfDimensions: Int
      def allocateStackBuffers(stack: MemoryStack): NDimensionBuffer
    }

    implicit final class GlobalAndLocalDimensionBufferAllocator(dimensions: Iterable[GlobalAndLocalDimension])
        extends NDimensionBufferAllocator {
      def this(dimensions: GlobalAndLocalDimension*) = this(dimensions)
      override def allocateStackBuffers(stack: MemoryStack): NDimensionBuffer = {
        val globalWorkOffsetBuffer = stack.mallocPointer(numberOfDimensions)
        val globalWorkSizeBuffer = stack.mallocPointer(numberOfDimensions)
        val localWorkSizeBuffer = stack.mallocPointer(numberOfDimensions)
        for ((dimension, i) <- dimensions.view.zipWithIndex) {
          globalWorkOffsetBuffer.put(i, dimension.globalWorkOffset.toLong)
          globalWorkSizeBuffer.put(i, dimension.globalWorkSize.toLong)
          localWorkSizeBuffer.put(i, dimension.localWorkSize.toLong)
        }
        NDimensionBuffer(globalWorkOffsetBuffer, globalWorkSizeBuffer, localWorkSizeBuffer)
      }

      override val numberOfDimensions: Int = dimensions.size
    }

    implicit final class GlobalDimensionBufferAllocator(dimensions: Iterable[GlobalDimension])
        extends NDimensionBufferAllocator {
      def this(dimensions: GlobalDimension*) = this(dimensions)
      override def allocateStackBuffers(stack: MemoryStack): NDimensionBuffer = {
        val globalWorkOffsetBuffer = stack.mallocPointer(numberOfDimensions)
        val globalWorkSizeBuffer = stack.mallocPointer(numberOfDimensions)
        for ((dimension, i) <- dimensions.view.zipWithIndex) {
          globalWorkOffsetBuffer.put(i, dimension.globalWorkOffset.toLong)
          globalWorkSizeBuffer.put(i, dimension.globalWorkSize.toLong)
        }
        NDimensionBuffer(globalWorkOffsetBuffer, globalWorkSizeBuffer, null)
      }
      override val numberOfDimensions: Int = dimensions.size
    }

    implicit final class GlobalWorkSizeOnlyDimensionBufferAllocator(dimensions: Iterable[GlobalWorkSizeOnlyDimension])
        extends NDimensionBufferAllocator {
      def this(dimensions: GlobalWorkSizeOnlyDimension*) = this(dimensions)
      override def allocateStackBuffers(stack: MemoryStack): NDimensionBuffer = {
        val globalWorkSizeBuffer = stack.mallocPointer(numberOfDimensions)
        for ((dimension, i) <- dimensions.view.zipWithIndex) {
          globalWorkSizeBuffer.put(i, dimension.globalWorkSize.toLong)
        }
        NDimensionBuffer(null, globalWorkSizeBuffer, null)
      }
      override val numberOfDimensions: Int = dimensions.size
    }

    final case class GlobalWorkSizeOnlyDimension(globalWorkSize: Address)
    final case class GlobalDimension(globalWorkOffset: Address, globalWorkSize: Address)
    final case class GlobalAndLocalDimension(globalWorkOffset: Address,
                                             globalWorkSize: Address,
                                             localWorkSize: Address)
  }

  final class CommandQueue private[OpenCL] (val handle: Address)
      extends AssertionAutoCloseable
      with AssertionFinalizer {
    import CommandQueue._
    def duplicate(): CommandQueue = {
      checkErrorCode(clRetainCommandQueue(handle.toLong))
      new CommandQueue(handle)
    }

    override protected def forceClose(): Unit = {
      checkErrorCode(clReleaseCommandQueue(handle.toLong))
    }

    def enqueueNDRangeKernel(kernel: Kernel,
                             dimensions: NDimensionBufferAllocator,
                             preconditionEvents: Event*): Event = {
      val stack = stackPush()
      val outputEvent = try {
        val inputEventBuffer = if (preconditionEvents.isEmpty) {
          null
        } else {
          stack.pointers(preconditionEvents.view.map(_.handle.toLong): _*)
        }
        val outputEventBuffer = stack.pointers(0L)
        val ndimensionBuffer = dimensions.allocateStackBuffers(stack)
        checkErrorCode(
          clEnqueueNDRangeKernel(
            handle.toLong,
            kernel.handle.toLong,
            dimensions.numberOfDimensions,
            ndimensionBuffer.globalWorkOffset,
            ndimensionBuffer.globalWorkSize,
            ndimensionBuffer.localWorkSize,
            inputEventBuffer,
            outputEventBuffer
          )
        )
        outputEventBuffer.get(0)
      } finally {
        stack.close()
      }
      checkErrorCode(clFlush(handle.toLong))
      new Event(Address(outputEvent))
    }

    def enqueueReadBuffer[Element, Destination](
        source: Buffer[Element],
        destination: Destination,
        preconditionEvents: Event*)(implicit memory: Memory.Aux[Element, Destination]): Event = {
      val readBufferEvent = {
        val stack = stackPush()
        try {
          val (inputEventBufferSize, inputEventBufferAddress) = if (preconditionEvents.isEmpty) {
            (0, NULL)
          } else {
            val inputEventBuffer = stack.pointers(preconditionEvents.view.map(_.handle.toLong): _*)
            (preconditionEvents.length, inputEventBuffer.address())
          }
          val outputEventBuffer = stack.pointers(0L)
          checkErrorCode(
            nclEnqueueReadBuffer(
              handle.toLong,
              source.handle.toLong,
              CL_FALSE,
              0,
              memory.remainingBytes(destination),
              memory.address(destination).toLong,
              inputEventBufferSize,
              inputEventBufferAddress,
              outputEventBuffer.address()
            )
          )
          outputEventBuffer.get(0)
        } finally {
          stack.close()
        }
      }
      checkErrorCode(clFlush(handle.toLong))
      new Event(Address(readBufferEvent))
    }

  }

  final class Event(val handle: Address) extends AssertionAutoCloseable with AssertionFinalizer {

    def duplicate(): Event = {
      checkErrorCode(clRetainEvent(handle.toLong))
      new Event(handle)
    }

    override protected def forceClose(): Unit = {
      checkErrorCode(clReleaseEvent(handle.toLong))
    }

    def commandExecutionStatus: Int = {
      val a = Array(0)
      checkErrorCode(clGetEventInfo(handle.toLong, CL_EVENT_COMMAND_EXECUTION_STATUS, a, null))
      a(0)
    }

    def waitFor(callbackType: Int): Task[Unit] = Task.async { handler =>
      object Callback extends CLEventCallbackI {
        override final def invoke(event: Long, status: Int, userData: Long): Unit = {
          container.close()
          status match {
            case `callbackType` => handler(\/-(()))
            case errorCode if errorCode < 0 => handler(-\/(Exceptions.fromErrorCode(errorCode)))
            case _ => throw new IllegalStateException()
          }
        }
        val container: CLEventCallback = CLEventCallback.create(this)
      }
      checkErrorCode(
        clSetEventCallback(
          handle.toLong,
          callbackType,
          Callback.container,
          NULL
        )
      )
    }

    def waitForComplete(): Task[Unit] = waitFor(CL_COMPLETE)

  }

  object Buffer {

    implicit def bufferBox[Element, BufferElemen]: Box.Aux[Buffer[Element], Address] = new Box[Buffer[Element]] {
      override type Raw = Address

      override def box(raw: Raw): Buffer[Element] =
        new Buffer[Element](raw)

      override def unbox(boxed: Buffer[Element]): Raw = boxed.handle
    }
  }

  // TODO: remove the `Element` type parameter
  final class Buffer[Element] private[OpenCL] (val handle: Address)
      extends AssertionAutoCloseable
      with AssertionFinalizer {

    def numberOfBytes: Int = {
      val sizeBuffer: Array[Long] = Array(0L)
      checkErrorCode(clGetMemObjectInfo(handle.toLong, CL_MEM_SIZE, sizeBuffer, null))
      val Array(value) = sizeBuffer
      if (value.isValidInt) {
        value.toInt
      } else {
        throw new IllegalStateException(s"Buffer's numberOfBytes($value) is too large")
      }
    }

    def length(implicit memory: Memory[Element]): Int = numberOfBytes / memory.numberOfBytesPerElement

    def duplicate(): Buffer[Element] = {
      checkErrorCode(clRetainMemObject(handle.toLong))
      new Buffer(handle)
    }

    override protected def forceClose(): Unit = {
      checkErrorCode(clReleaseMemObject(handle.toLong))
    }
  }

  object Program {
    private[Program] final class ManagedCallback(handler: Unit => Unit) extends CLProgramCallbackI {
      override def invoke(program: Long, user_data: Long): Unit = {
        container.close()
        handler(())
      }
      val container: CLProgramCallback = CLProgramCallback.create(this)
    }
  }

  final class Program private[OpenCL] (val handle: Address) extends AssertionAutoCloseable with AssertionFinalizer {

    def duplicate(): Program = {
      checkErrorCode(clRetainProgram(handle.toLong))
      new Program(handle)
    }

    override protected def forceClose(): Unit = {
      checkErrorCode(clReleaseProgram(handle.toLong))
    }

    def build(devices: Seq[Device], options: CharSequence = ""): Future[Unit] = Future.async { (handler: Unit => Unit) =>
      val callbackContainer = new Program.ManagedCallback(handler).container
      val stack = stackPush()
      try {
        checkErrorCode(
          clBuildProgram(handle.toLong, stack.pointers(devices.map(_.id): _*), options, callbackContainer, NULL))
      } finally {
        stack.close()
      }
    }

    def build(options: CharSequence): Future[Unit] = Future.async { (handler: Unit => Unit) =>
      val callback = new Program.ManagedCallback(handler)
      checkErrorCode(clBuildProgram(handle.toLong, null, options, callback.container, NULL))
    }

    def build(): Future[Unit] = build("")

    def createKernel(kernelName: CharSequence): Kernel = {
      val errorCodeBuffer = Array(0)
      val kernelHandle = clCreateKernel(handle.toLong, kernelName, errorCodeBuffer)
      checkErrorCode(errorCodeBuffer(0))
      new Kernel(Address(kernelHandle))
    }

  }

  final class Kernel private[OpenCL] (val handle: Address) extends AssertionAutoCloseable with AssertionFinalizer {

    def setArg[A](argIndex: Int, a: A)(implicit memory: Memory[A]): Unit = {
      val stack = stackPush()
      try {
        val byteBuffer = stack.malloc(memory.numberOfBytesPerElement)
        memory.put(memory.fromByteBuffer(byteBuffer), 0, a)
        checkErrorCode(nclSetKernelArg(handle.toLong, argIndex, byteBuffer.remaining, memAddress(byteBuffer)))
      } finally {
        stack.close()
      }

    }

    def duplicate(): Kernel = {
      checkErrorCode(clRetainKernel(handle.toLong))
      new Kernel(handle)
    }

    override protected def forceClose(): Unit = {
      checkErrorCode(clReleaseKernel(handle.toLong))
    }
  }

}
