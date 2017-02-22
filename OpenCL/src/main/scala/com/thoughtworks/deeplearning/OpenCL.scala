package com.thoughtworks.deeplearning

import java.io.Closeable
import java.nio.{ByteBuffer, IntBuffer}

import org.lwjgl.opencl._
import CL10._
import CL11._
import CL12._
import org.lwjgl.BufferUtils
import org.lwjgl.system.MemoryUtil._
import org.lwjgl.system.MemoryStack._
import org.lwjgl.system.Pointer._

import scala.collection.mutable
import com.qifun.statelessFuture.Future

import scala.util.control.Exception.Catcher
import scala.util.control.TailCalls
import scala.util.control.TailCalls.TailRec
import scala.util.{Failure, Success, Try}

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

  final case class Device(id: Long, capabilities: CLCapabilities) {

    def longInfo(paramName: Int): Long = {
      val buffer = Array[Long](0L)
      checkErrorCode(clGetDeviceInfo(id, paramName, buffer, null))
      val Array(value) = buffer
      value
    }

    /**
      * Describes the command-queue properties supported by the device.
      * @see [[CL_DEVICE_QUEUE_PROPERTIES]]
      * @return
      */
    def queueProperties: Long = longInfo(CL_DEVICE_QUEUE_PROPERTIES)

    def deviceType: Long = longInfo(CL_DEVICE_TYPE)
  }

  final case class Platform(id: Long, capabilities: CLCapabilities) {

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
        val callbackContainer = CLContextCallback.create(new ManagedContextCallbackI(logger))
        val context = clCreateContext(contextProperties, deviceIds, callbackContainer, NULL, errorCodeBuffer)
        checkErrorCode(errorCodeBuffer.get(0))
        new Context(context, callbackContainer)
      } finally {
        stack.close()
      }
    }

  }

  final class ManagedContextCallbackI(logger: (String, ByteBuffer) => Unit) extends CLContextCallbackI {
    override def invoke(errInfo: Long, privateInfo: Long, size: Long, user_data: Long): Unit = {
      if (size.isValidInt) {
        logger(memASCII(errInfo), memByteBuffer(privateInfo, size.toInt))
      } else {
        throw new IllegalArgumentException(s"numberOfBytes($size) is too large")
      }
    }
  }

  trait SizeOf[T] {
    // TODO: add encoding / decoding methods
    def numberOfBytes: Int
  }

  object SizeOf {

    implicit object ByteSizeOf extends SizeOf[Byte] {
      override def numberOfBytes: Int = java.lang.Byte.BYTES
    }

    implicit object ShortSizeOf extends SizeOf[Short] {
      override def numberOfBytes: Int = java.lang.Short.BYTES
    }

    implicit object IntSizeOf extends SizeOf[Int] {
      override def numberOfBytes: Int = java.lang.Integer.BYTES
    }

    implicit object LongSizeOf extends SizeOf[Long] {
      override def numberOfBytes: Int = java.lang.Long.BYTES
    }

    implicit object CharSizeOf extends SizeOf[Char] {
      override def numberOfBytes: Int = java.lang.Character.BYTES
    }

    implicit object FloatSizeOf extends SizeOf[Float] {
      override def numberOfBytes: Int = java.lang.Float.BYTES
    }

    implicit object DoubleSizeOf extends SizeOf[Double] {
      override def numberOfBytes: Int = java.lang.Double.BYTES
    }

  }

  final class Context(val handle: Long, callback: CLContextCallback) extends Releasable {

    def createCommandQueue(device: Device, properties: Long): CommandQueue = {
      val a = Array(0)
      val commandQueue = clCreateCommandQueue(handle, device.id, properties, a);
      checkErrorCode(a(0))
      new CommandQueue(commandQueue)

    }

    def duplicate(): Context = {
      checkErrorCode(clRetainContext(handle))
      new Context(handle, callback)
    }

    override protected def release(): Unit = {
      val rcBuffer = Array(0)
      checkErrorCode(clGetContextInfo(handle, CL_CONTEXT_REFERENCE_COUNT, rcBuffer, null))
      rcBuffer match {
        case Array(1) =>
          // It's the last reference
          checkErrorCode(clReleaseContext(handle))
          callback.close()
        case _ =>
          checkErrorCode(clReleaseContext(handle))
      }
    }

    def createBuffer[Element](size: Long)(implicit sizeOf: SizeOf[Element]): Buffer[Element] = {
      val stack = stackPush()
      try {
        val errorCodeBuffer = stack.ints(0)
        val buffer = clCreateBuffer(handle, CL_MEM_READ_WRITE, sizeOf.numberOfBytes * size, errorCodeBuffer)
        checkErrorCode(errorCodeBuffer.get(0))
        new Buffer(buffer)
      } finally {
        stack.pop()
      }
    }

  }

  final class CommandQueue(val handle: Long) extends Releasable {

    def duplicate(): CommandQueue = {
      checkErrorCode(clRetainCommandQueue(handle))
      new CommandQueue(handle)
    }

    override protected def release(): Unit = {
      checkErrorCode(clReleaseCommandQueue(handle))
    }

    def readRaw[Element](buffer: Buffer[Element], preconditionEvents: Event[_]*)(
        implicit sizeOf: SizeOf[Element]): ReadBuffer = {

      val output: ByteBuffer = BufferUtils.createByteBuffer(buffer.numberOfBytes)
      val readBufferEvent = {
        val stack = stackPush()
        try {
          val inputEventBuffer = if (preconditionEvents.isEmpty) {
            null
          } else {
            stack.pointers(preconditionEvents.view.map(_.handle): _*)
          }
          val outputEventBuffer = stack.pointers(0L)
          checkErrorCode(
            clEnqueueReadBuffer(handle, buffer.handle, CL_FALSE, 0, output, inputEventBuffer, outputEventBuffer))
          outputEventBuffer.get(0)
        } finally {
          stack.close()
        }
      }
      checkErrorCode(clFlush(handle))
      new ReadBuffer(readBufferEvent, output)
    }

  }

  final class ReadBuffer(override val handle: Long, protected val result: ByteBuffer) extends Event[ByteBuffer] {
    def duplicate(): ReadBuffer = {
      checkErrorCode(clRetainEvent(handle))
      new ReadBuffer(handle, result)
    }
  }

  trait Event[Result] extends Releasable with Future.Stateful[Result] {
    val handle: Long

    override protected def release(): Unit = {
      checkErrorCode(clReleaseEvent(handle))
    }

    final protected def commandExecutionStatus: Int = {
      val a = Array(0)
      checkErrorCode(clGetEventInfo(handle, CL_EVENT_COMMAND_EXECUTION_STATUS, a, null))
      a(0)
    }

    protected def result: Result

    override def value: Option[Try[Result]] = {
      commandExecutionStatus match {
        case CL_QUEUED => None
        case CL_SUBMITTED => None
        case CL_RUNNING => None
        case CL_COMPLETE => Some(Success(result))
        case errorCode => Some(Failure(Exceptions.fromErrorCode(errorCode)))
      }
    }

    override def onComplete(handler: Result => TailRec[Unit])(
        implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
      object Callback extends CLEventCallbackI {
        override final def invoke(event: Long, status: Int, user_data: Long): Unit = {
          container.close()
          val Some(resultOrException) = value
          (resultOrException match {
            case Success(result) =>
              handler(result)
            case Failure(exception) =>
              catcher(exception)
          }).result
        }
        val container: CLEventCallback = CLEventCallback.create(this)
      }
      checkErrorCode(
        clSetEventCallback(
          handle,
          CL_COMPLETE,
          Callback.container,
          NULL
        )
      )
      TailCalls.done(())
    }
  }

  final class Buffer[Element](val handle: Long) extends Releasable {

    def numberOfBytes: Int = {
      val sizeBuffer: Array[Long] = Array(0L)
      checkErrorCode(clGetMemObjectInfo(handle, CL_MEM_SIZE, sizeBuffer, null))
      val Array(value) = sizeBuffer
      if (value.isValidInt) {
        value.toInt
      } else {
        throw new IllegalStateException(s"Buffer's numberOfBytes($value) is too large")
      }
    }

    def duplicate(): Buffer[Element] = {
      checkErrorCode(clRetainMemObject(handle))
      new Buffer(handle)
    }

    override protected def release(): Unit = {
      checkErrorCode(clReleaseMemObject(handle))
    }
  }

}
