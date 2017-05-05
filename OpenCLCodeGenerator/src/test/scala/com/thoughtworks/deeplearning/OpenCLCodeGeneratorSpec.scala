package com.thoughtworks.deeplearning

import java.nio.{ByteBuffer, FloatBuffer, IntBuffer}

import com.thoughtworks.deeplearning.OpenCLCodeGenerator._
import org.lwjgl.opencl._
import CL10._
import CL11._
import CL12._
import org.scalatest.{FreeSpec, Matchers}
import org.lwjgl.BufferUtils
import org.lwjgl.system.MemoryUtil._
import org.lwjgl.system.MemoryStack._
import org.lwjgl.system.Pointer._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class OpenCLCodeGeneratorSpec extends FreeSpec with Matchers {

  private def checkCLError(errorCode: Int) = {
    if (errorCode != CL_SUCCESS) {
      throw new IllegalStateException(raw"""OpenCL error [$errorCode]""")
    }
  }

  val PlatformIndex = 0
  val DeviceIndex = 0

  def platformRank(platformId: Long, platformCapabilities: CLCapabilities): Unit = {}
  def deviceRank(deviceId: Long, deviceCapabilities: CLCapabilities): Unit = {}

  "Plus" in {

    val kernel = OpenCLCodeGenerator.KernelDefinition(
      "f",
      Seq(Parameter('output, DslType.DslBuffer(DslType.DslStructure(List(DslType.DslDouble))))),
      Seq(
        DslEffect.Update(
          DslExpression.Identifier('output),
          DslExpression.GetGlobalId(DslExpression.IntLiteral(0)),
          DslExpression.HCons(
            DslExpression.Plus(DslExpression.DoubleLiteral(1.5), DslExpression.DoubleLiteral(1.5), DslType.DslDouble),
            DslExpression.HNilLiteral),
          DslType.DslStructure(List(DslType.DslDouble))
        ))
    )
    val cl = OpenCLCodeGenerator.generateSourceCode(kernel).toArray[CharSequence]
    cl should not be empty
//    println(cl.mkString)
    val output = Array(0.0)

    val stack = stackPush()
    try {

      val Array(numberOfPlatformIDs) = {
        val a = Array(0)
        checkCLError(clGetPlatformIDs(null, a))
        a
      }
      val platformIds = stack.mallocPointer(numberOfPlatformIDs)
      checkCLError(clGetPlatformIDs(platformIds, null: IntBuffer))

      val (platformId, platformCapabilities) = (for (i <- 0 until platformIds.capacity) yield {
        val platformId = platformIds.get(i)
        val platformCapabilities = CL.createPlatformCapabilities(platformId)
        platformId -> platformCapabilities
      }).maxBy((platformRank _).tupled)

      val Array(numberOfDevices) = {
        val a = Array(0)
        checkCLError(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, null, a))
        a
      }
      val deviceIds = stack.mallocPointer(numberOfDevices)
      checkCLError(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, deviceIds, null: IntBuffer))

      val (deviceId, deviceCapabilities) = (for (i <- 0 until deviceIds.capacity()) yield {
        val deviceId = deviceIds.get(i)
        val deviceCapabilities = CL.createDeviceCapabilities(deviceId, platformCapabilities)
        deviceId -> deviceCapabilities
      }).maxBy((deviceRank _).tupled)

      val callback = CLContextCallback.create(new CLContextCallbackI {
        override def invoke(errInfo: Long, privateInfo: Long, size: Long, userData: Long): Unit = {
          println(memASCII(errInfo))
          memByteBuffer(privateInfo, size.toInt)

        }
      })
      try {

        stack.push()
        val context = try {
          val errorCode = stack.ints(0)
          val contextProperties = stack.pointers(CL_CONTEXT_PLATFORM, platformId, 0)
          val context = clCreateContext(contextProperties, deviceId, callback, NULL, errorCode)
          checkCLError(errorCode.get(0))
          context
        } finally {
          stack.pop()
        }
        try {
          val commandQueue = {
            val a = Array(0)
            val commandQueue = clCreateCommandQueue(context, deviceId, NULL, a);
            checkCLError(a.head)
            commandQueue
          }
          try {
            stack.push()
            val program = try {
              val errorCode = stack.ints(0)
              val program = clCreateProgramWithSource(context, cl, errorCode)
              checkCLError(errorCode.get(0))
              program
            } finally {
              stack.pop()
            }
            try {

              checkCLError(clBuildProgram(program, deviceId, "", null, NULL))

              val kernel = {
                val a = Array(0)
                val kernel = clCreateKernel(program, "f", a)
                checkCLError(a(0))
                kernel
              }
              try {
                // TODO: 此处只为调试
                val buffer = {
                  val a = Array(0)
                  val buffer = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, Array(0.0), a)
                  checkCLError(a(0))
                  buffer
                }
                try {
                  checkCLError(clSetKernelArg1p(kernel, 0, buffer))
                  stack.push()
                  val event = try {
                    val eventPointer = stack.pointers(0L)
                    checkCLError(
                      clEnqueueNDRangeKernel(commandQueue,
                                             kernel,
                                             1,
                                             stack.pointers(0L),
                                             stack.pointers(1L),
                                             stack.pointers(1L),
                                             null,
                                             eventPointer))
                    eventPointer.get(0)
                  } finally {
                    stack.pop()
                  }

                  val event2 = try {
                    stack.push()
                    try {
                      val eventPointer1 = stack.pointers(event)
                      val eventPointer2 = stack.pointers(0L)
                      checkCLError(
                        clEnqueueReadBuffer(commandQueue, buffer, CL_FALSE, 0, output, eventPointer1, eventPointer2))
                      eventPointer2.get(0)
                    } finally {
                      stack.pop()
                    }
                  } finally {
                    clReleaseEvent(event)

                  }
//                  checkCLError(
//                    clSetEventCallback(
//                      event,
//                      CL_COMPLETE,
//                      new CLEventCallbackI {
//                        override def invoke(event2: Long, status: Int, user_data: Long): Unit = {
//                          println(s"error $status")
//                        }
//                      },
//                      NULL
//                    ))
                  checkCLError(
                    clSetEventCallback(
                      event2,
                      CL_COMPLETE,
                      new CLEventCallbackI {
                        override def invoke(event2: Long, status: Int, user_data: Long): Unit = {
                          clReleaseEvent(event2)
                        }
                      },
                      NULL
                    ))

                } finally {
                  clReleaseMemObject(buffer)
                }

              } finally {
                clReleaseKernel(kernel)
              }
            } finally {
              clReleaseProgram(program)
            }
          } finally {
            checkCLError(clReleaseCommandQueue(commandQueue))
          }
        } finally {
          checkCLError(clReleaseContext(context))
        }
      } finally {
        callback.close()
      }

    } finally {
      stack.close()
    }

    output should be(Array(3.0))
  }
}
