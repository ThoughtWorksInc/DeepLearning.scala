package com.thoughtworks.deeplearning

import java.io.Closeable

import org.lwjgl.opencl._
import CL10._
import CL11._
import CL12._
import org.lwjgl.system.MemoryUtil
import org.lwjgl.opencl._
import CL10._
import CL11._
import CL12._
import org.lwjgl.BufferUtils
import org.lwjgl.system.MemoryUtil._
import org.lwjgl.system.MemoryStack._
import org.lwjgl.system.Pointer._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object OpenCL {

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
