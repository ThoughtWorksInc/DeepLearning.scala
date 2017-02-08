package com.thoughtworks.deeplearning

import java.io.Closeable

import org.lwjgl.opencl._
import CL10._
import CL11._
import CL12._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class OpenCLBufferFactory(clContext: Long, clCommandQueue: Long) extends Closeable {

  final class Buffer(clMem: Long) extends Closeable {

    private var isClosed = false

    override def close(): Unit = synchronized {
      if (!isClosed) {
        isClosed = true
        clReleaseMemObject(clMem) match {
          case CL_SUCCESS =>
          case errorCode =>
            throw new IllegalStateException(s"clReleaseMemObject error: $errorCode")
        }
      }
    }

    override def finalize(): Unit = {
      close()
    }

  }

  private var isClosed = false

  override def close(): Unit = synchronized {
    if (!isClosed) {
      isClosed = true
      (clReleaseCommandQueue(clCommandQueue), clReleaseContext(clContext)) match {
        case (CL_SUCCESS, CL_SUCCESS) =>
        case (errorCode, CL_SUCCESS) =>
          throw new IllegalStateException(s"clReleaseCommandQueue error: $errorCode")
        case (CL_SUCCESS, errorCode) =>
          throw new IllegalStateException(s"clReleaseContext error: $errorCode")
        case (commandQueueErrorCode, contextErrorCode) =>
          throw new IllegalStateException(raw"""clReleaseCommandQueue error: $commandQueueErrorCode.
clReleaseContext error: $contextErrorCode""")

      }
    }
  }

  override def finalize(): Unit = {
    close()
  }

}
