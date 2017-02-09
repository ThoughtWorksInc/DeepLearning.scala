package com.thoughtworks.deeplearning

import java.io.Closeable

import org.lwjgl.opencl._
import CL10._
import CL11._
import CL12._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object OpenCL {

  final class Context(clContext: Long) extends Releasable {

    def duplicate(): Context = {
      clRetainContext(clContext) match {
        case CL_SUCCESS =>
          new Context(clContext)
        case errorCode =>
          throw new IllegalStateException(s"clRetainContext error: $errorCode")
      }
    }

    override protected def release(): Unit = {
      clReleaseContext(clContext) match {
        case CL_SUCCESS =>
        case errorCode =>
          throw new IllegalStateException(s"clReleaseContext error: $errorCode")
      }
    }

  }

  final class CommandQueue(clCommandQueue: Long) extends Releasable {

    def duplicate(): CommandQueue = {
      clRetainCommandQueue(clCommandQueue) match {
        case CL_SUCCESS =>
          new CommandQueue(clCommandQueue)
        case errorCode =>
          throw new IllegalStateException(s"clRetainCommandQueue error: $errorCode")
      }
    }

    override protected def release(): Unit = {
      clReleaseCommandQueue(clCommandQueue) match {
        case CL_SUCCESS =>
        case errorCode =>
          throw new IllegalStateException(s"clReleaseCommandQueue error: $errorCode")
      }
    }

  }

  final class Buffer(clMem: Long) extends Releasable {

    def duplicate(): Buffer = {
      clRetainMemObject(clMem) match {
        case CL_SUCCESS =>
          new Buffer(clMem)
        case errorCode =>
          throw new IllegalStateException(s"clRetainMemObject error: $errorCode")
      }
    }

    override protected def release(): Unit = {
      clReleaseMemObject(clMem) match {
        case CL_SUCCESS =>
        case errorCode =>
          throw new IllegalStateException(s"clReleaseMemObject error: $errorCode")
      }
    }
  }

}
