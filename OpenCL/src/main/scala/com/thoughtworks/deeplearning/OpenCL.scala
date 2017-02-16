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

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object OpenCL {

  private object ContextCallbackDispatcher {

    private val managedCallbacksById = new mutable.WeakHashMap[Long, (String, ByteBuffer) => Unit]
    private var seed = 0L

    def register(managedCallback: (String, ByteBuffer) => Unit): Long = {
      val id = seed
      seed += 1
      managedCallbacksById.put(id, managedCallback)
      id
    }

    val callback: CLContextCallback = CLContextCallback.create(new CLContextCallbackI {
      override def invoke(errInfo: Long, privateInfo: Long, size: Long, userData: Long): Unit = {
        managedCallbacksById(userData).apply(memASCII(errInfo), memByteBuffer(privateInfo, size.toInt))
      }
    })

    override protected def finalize(): Unit = {
      callback.close()
    }

  }

  def platforms: Seq[Platform] = {
    val Array(numberOfPlatformIDs) = {
      val a = Array(0)
      clGetPlatformIDs(null, a) match {
        case CL_SUCCESS =>
          a
        case errorCode =>
          throw new IllegalStateException(s"clGetPlatformIDs error: $errorCode")
      }
    }
    val stack = stackPush()
    try {
      val platformIdBuffer = stack.mallocPointer(numberOfPlatformIDs)
      clGetPlatformIDs(platformIdBuffer, null: IntBuffer) match {
        case CL_SUCCESS =>
          for (i <- (0 until platformIdBuffer.capacity).view) yield {
            val platformId = platformIdBuffer.get(i)
            val platformCapabilities = CL.createPlatformCapabilities(platformId)
            Platform(platformId, platformCapabilities)
          }
        case errorCode =>
          throw new IllegalStateException(s"clGetPlatformIDs error: $errorCode")
      }
    } finally {
      stack.close()
    }
  }

  final case class Device(id: Long, capabilities: CLCapabilities)

  final case class Platform(id: Long, capabilities: CLCapabilities) {

    def cpus: Seq[Device] = devicesByType(CL_DEVICE_TYPE_CPU)

    def gpus: Seq[Device] = devicesByType(CL_DEVICE_TYPE_GPU)

    def accelerators: Seq[Device] = devicesByType(CL_DEVICE_TYPE_ACCELERATOR)

    def devices: Seq[Device] = devicesByType(CL_DEVICE_TYPE_ALL)

    def devicesByType(deviceType: Int): Seq[Device] = {
      val Array(numberOfDevices) = {
        val a = Array(0)
        clGetDeviceIDs(Platform.this.id, deviceType, null, a) match {
          case CL_SUCCESS =>
            a
          case errorCode =>
            throw new IllegalStateException(s"clGetDeviceIDs error: $errorCode")
        }
      }
      val stack = stackPush()
      try {

        val deviceIds = stack.mallocPointer(numberOfDevices)
        clGetDeviceIDs(Platform.this.id, deviceType, deviceIds, null: IntBuffer) match {
          case CL_SUCCESS =>
            for (i <- 0 until deviceIds.capacity()) yield {
              val deviceId = deviceIds.get(i)
              val deviceCapabilities = CL.createDeviceCapabilities(deviceId, Platform.this.capabilities)
              Device(deviceId, deviceCapabilities)
            }
          case errorCode =>
            throw new IllegalStateException(s"clGetDeviceIDs error: $errorCode")
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
        val context = clCreateContext(contextProperties,
                                      deviceIds,
                                      ContextCallbackDispatcher.callback,
                                      ContextCallbackDispatcher.register(logger),
                                      errorCodeBuffer)
        errorCodeBuffer.get(0) match {
          case CL_SUCCESS => new Context(context)
          case errorCode => throw new IllegalStateException(s"clCreateContext error: $errorCode")

        }
      } finally {
        stack.close()
      }
    }

  }

  trait SizeOf[T] {
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

  final class Context(val handle: Long) extends Releasable {

    def createCommandQueue(device: Device, properties: Long): CommandQueue = {
      val a = Array(0)
      val commandQueue = clCreateCommandQueue(handle, device.id, properties, a);
      a(0) match {
        case CL_SUCCESS =>
          new CommandQueue(commandQueue)
        case errorCode =>
          throw new IllegalStateException(s"clCreateCommandQueue error: $errorCode")
      }
    }

    def duplicate(): Context = {
      clRetainContext(handle) match {
        case CL_SUCCESS =>
          new Context(handle)
        case errorCode =>
          throw new IllegalStateException(s"clRetainContext error: $errorCode")
      }
    }

    override protected def release(): Unit = {
      clReleaseContext(handle) match {
        case CL_SUCCESS =>
        case errorCode =>
          throw new IllegalStateException(s"clReleaseContext error: $errorCode")
      }
    }

    def createBuffer[Element](size: Long)(implicit sizeOf: SizeOf[Element]): Buffer[Element] = {
      val stack = stackPush()
      try {
        val errorCodeBuffer = stack.ints(0)
        val buffer = clCreateBuffer(handle, CL_MEM_READ_WRITE, sizeOf.numberOfBytes * size, errorCodeBuffer)
        errorCodeBuffer.get(0) match {
          case CL_SUCCESS =>
          case errorCode =>
            throw new IllegalStateException(s"clCreateBuffer error: $errorCode")
        }
        new Buffer(buffer)
      } finally {
        stack.pop()
      }
    }

  }

  final class CommandQueue(val handle: Long) extends Releasable {

    def duplicate(): CommandQueue = {
      clRetainCommandQueue(handle) match {
        case CL_SUCCESS =>
          new CommandQueue(handle)
        case errorCode =>
          throw new IllegalStateException(s"clRetainCommandQueue error: $errorCode")
      }
    }

    override protected def release(): Unit = {
      clReleaseCommandQueue(handle) match {
        case CL_SUCCESS =>
        case errorCode =>
          throw new IllegalStateException(s"clReleaseCommandQueue error: $errorCode")
      }
    }

  }

  final class Buffer[Element](val handle: Long) extends Releasable {

    def duplicate(): Buffer[Element] = {
      clRetainMemObject(handle) match {
        case CL_SUCCESS =>
          new Buffer(handle)
        case errorCode =>
          throw new IllegalStateException(s"clRetainMemObject error: $errorCode")
      }
    }

    override protected def release(): Unit = {
      clReleaseMemObject(handle) match {
        case CL_SUCCESS =>
        case errorCode =>
          throw new IllegalStateException(s"clReleaseMemObject error: $errorCode")
      }
    }
  }

}
