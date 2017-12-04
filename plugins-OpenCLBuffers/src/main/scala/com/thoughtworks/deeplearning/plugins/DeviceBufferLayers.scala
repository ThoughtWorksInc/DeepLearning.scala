package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.compute.{Memory, OpenCL}
import com.thoughtworks.continuation.UnitContinuation
import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.raii.asynchronous._
import com.thoughtworks.continuation._
import com.thoughtworks.deeplearning.plugins.Layers._
import com.thoughtworks.future._
import shapeless.Witness
import DeepLearning.ops._
import scalaz.syntax.all._

import scalaz.Tags.Parallel
import scalaz.{Apply, Semigroup}

trait DeviceBufferLayers extends Layers with OpenCL {

  trait ImplicitsApi extends super[Layers].ImplicitsApi with super[OpenCL].ImplicitsApi

  override type Implicits <: ImplicitsApi

  trait DeviceBufferLayerApi extends LayerApi {

    override type Data = DeviceBuffer[Element]
    override type Delta = DeviceBuffer[Element]
    type Element
  }

  type DeviceBufferLayer <: DeviceBufferLayerApi with Layer

  def mean[Operand0, Buffer, Element, OutputLayer](operand0: Operand0)(
      implicit
      deepLearning0: DeepLearning.Aux[Operand0, Buffer, Buffer],
      isDoBuffer: Do[Tape[Buffer, Buffer]] <:< Do[Tape[DeviceBuffer[Element], DeviceBuffer[Element]]],
      layerFactory: ToLayer.Aux[Element, Element, OutputLayer]): OutputLayer = ???

  def matrixMultiply[Operand0, Operand1, Buffer, Element, OutputLayer /* <: DeviceBufferLayer */ ](operand0: Operand0,
                                                                                                   operand1: Operand1,
                                                                                                   matrix0Columns: Int)(
      implicit
      deepLearning0: DeepLearning.Aux[Operand0, Buffer, Buffer],
      deepLearning1: DeepLearning.Aux[Operand1, Buffer, Buffer],
      isDoBuffer: Do[Tape[Buffer, Buffer]] <:< Do[Tape[DeviceBuffer[Element], DeviceBuffer[Element]]],
      layerFactory: ToLayer.Aux[DeviceBuffer[Element], DeviceBuffer[Element], OutputLayer],
      memory: Memory[Element]
  ): OutputLayer = {

    val operand0Forward: Do[Tape[DeviceBuffer[Element], DeviceBuffer[Element]]] = isDoBuffer(operand0.forward)
    val operand1Forward: Do[Tape[DeviceBuffer[Element], DeviceBuffer[Element]]] = isDoBuffer(operand1.forward)

    val Parallel(tupledDo) = Apply[ParallelDo].tuple2(Parallel(operand0Forward), Parallel(operand1Forward))
    val forward = tupledDo.flatMap {
      case (Tape(data0, backward0), Tape(data1, backward1)) =>
        val data0Length = data0.length
        val data1Length = data1.length
        if (data0Length % matrix0Columns != 0) {
          throw new IllegalArgumentException("The data0 should be matrix")
        }

        val matrix0Rows = data0Length / matrix0Columns
        val matrix1Rows = matrix0Columns
        if (data1Length % matrix1Rows != 0) {
          throw new IllegalArgumentException("The data1 should be matrix")
        }

        val matrix1Columns = data1Length / matrix1Rows

        def outputData(data0: DeviceBuffer[Element], data1: DeviceBuffer[Element]): Do[DeviceBuffer[Element]] = {
          val doOutputBuffer = allocateBuffer[Element](matrix0Rows * matrix1Columns).flatMap {
            output: DeviceBuffer[Element] =>
              Do.monadicCloseable(matrixMultiplyProgram.createFirstKernel())
                .flatMap { kernel =>
                  kernel(0) = data0
                  kernel(1) = data1
                  kernel(2) = output
                  kernel(3) = matrix0Columns
                  kernel(4) = matrix1Columns
                  val self: this.type = this
                  val doEvent: Do[Event] = kernel.enqueue(matrix0Rows, matrix1Columns)(Witness(self))
                  doEvent.flatMap { event =>
                    val doWait: Do[Unit] = Do.garbageCollected(event.waitForComplete())
                    doWait
                  }
                }
                .intransitiveMap { _: Unit =>
                  output
                }
          }
          doOutputBuffer
        }

        def backward(doOutputDelta: Do[DeviceBuffer[Element]]): UnitContinuation[Unit] = {

          val delta0: Do[DeviceBuffer[Element]] = doOutputDelta.flatMap { outputDelta: DeviceBuffer[Element] =>
            if (outputDelta.length != matrix0Rows * matrix1Columns) {
              throw new IllegalArgumentException("The outputDelta should be matrix")
            }

            allocateBuffer[Element](matrix0Rows * matrix1Columns).flatMap { output: DeviceBuffer[Element] =>
              Do.monadicCloseable(backwardMatrixMultiplyProgram.createFirstKernel())
                .flatMap { kernel =>
                  kernel(0) = outputDelta
                  kernel(1) = data1
                  kernel(2) = output
                  kernel(3) = matrix1Columns
                  kernel(4) = matrix0Columns
                  val self: this.type = this
                  kernel.enqueue(matrix0Rows, matrix1Rows)(Witness(self)).flatMap { event =>
                    Do.garbageCollected(event.waitForComplete())
                  }
                }
                .intransitiveMap { _: Unit =>
                  output
                }
            }
          }
          val delta1: Do[DeviceBuffer[Element]] = doOutputDelta.flatMap { outputDelta: DeviceBuffer[Element] =>
            allocateBuffer[Element](matrix0Rows * matrix1Columns).flatMap { output: DeviceBuffer[Element] =>
              Do.monadicCloseable(backwardMatrixMultiplyProgram.createFirstKernel())
                .flatMap { kernel =>
                  kernel(0) = outputDelta
                  kernel(1) = data1
                  kernel(2) = output
                  kernel(3) = matrix1Columns
                  kernel(4) = matrix0Columns
                  val self: this.type = this
                  kernel.enqueue(matrix0Rows, matrix1Rows)(Witness(self)).flatMap { event =>
                    Do.garbageCollected(event.waitForComplete())
                  }
                }
                .intransitiveMap { _: Unit =>
                  output
                }
            }

          }
          backward0(delta0) >> backward1(delta1)
        }
        outputData(data0, data1).map(Tape(_, backward))
    }
    layerFactory.toLayer(forward)
  }

  def subtract[Buffer, Element, Operand0, Operand1, OutputLayer /* <: DeviceBufferLayer */ ](operand0: Operand0,
                                                                                             operand1: Operand1)(
      implicit
      deepLearning0: DeepLearning.Aux[Operand0, Buffer, Buffer],
      deepLearning1: DeepLearning.Aux[Operand1, Buffer, Buffer],
      isDoBuffer: Do[Tape[Buffer, Buffer]] <:< Do[Tape[DeviceBuffer[Element], DeviceBuffer[Element]]],
      layerFactory: ToLayer.Aux[DeviceBuffer[Element], DeviceBuffer[Element], OutputLayer],
      memory: Memory[Element]
  ): OutputLayer = {

    val operand0Forward: Do[Tape[DeviceBuffer[Element], DeviceBuffer[Element]]] = isDoBuffer(operand0.forward)
    val operand1Forward: Do[Tape[DeviceBuffer[Element], DeviceBuffer[Element]]] = isDoBuffer(operand1.forward)

    val Parallel(tupledDo) = Apply[ParallelDo].tuple2(Parallel(operand0Forward), Parallel(operand1Forward))

    val forward = tupledDo.flatMap {
      case (Tape(data0, backward0), Tape(data1, backward1)) =>
        val length = data0.length
        def outputData(data0: DeviceBuffer[Element], data1: DeviceBuffer[Element]): Do[DeviceBuffer[Element]] = {
          if (length != data1.length) {
            throw new IllegalArgumentException("The length of data0 should equal the length of data1")
          }
          val doOutputBuffer = allocateBuffer[Element](length).flatMap { output: DeviceBuffer[Element] =>
            Do.monadicCloseable(subtractProgram.createFirstKernel())
              .flatMap { kernel =>
                kernel(0) = data0
                kernel(1) = data1
                kernel(2) = output
                val self: this.type = this
                val doEvent: Do[Event] = kernel.enqueue(length)(Witness(self))
                doEvent.flatMap { event =>
                  val doWait: Do[Unit] = Do.garbageCollected(event.waitForComplete())
                  doWait
                }
              }
              .intransitiveMap { _: Unit =>
                output
              }
          }
          doOutputBuffer
        }

        def backward(doOutputDelta: Do[DeviceBuffer[Element]]): UnitContinuation[Unit] = {
          val delta0 = doOutputDelta
          val delta1: Do[DeviceBuffer[Element]] = doOutputDelta.flatMap { outputDelta: DeviceBuffer[Element] =>
            allocateBuffer[Element](length).flatMap { output: DeviceBuffer[Element] =>
              Do.monadicCloseable(negativeProgram.createFirstKernel())
                .flatMap { kernel =>
                  kernel(0) = outputDelta
                  kernel(1) = output
                  val self: this.type = this
                  kernel.enqueue(length)(Witness(self)).flatMap { event =>
                    Do.garbageCollected(event.waitForComplete())
                  }
                }
                .intransitiveMap { _: Unit =>
                  output
                }
            }
          }

          parallelAppend(backward0(delta0), backward1(delta1))
        }

        outputData(data0, data1).map(Tape(_, backward))
    }

    layerFactory.toLayer(forward)
  }

  private def parallelAppend(continuation0: UnitContinuation[Unit], continuation1: UnitContinuation[Unit]) = {
    val parallelContinuation0: ParallelContinuation[Unit] = Parallel(continuation0)
    val parallelContinuation1: ParallelContinuation[Unit] = Parallel(continuation1)
    import scalaz.std.anyVal._
    val Parallel(combined) =
      Semigroup
        .liftSemigroup[ParallelContinuation, Unit]
        .append(
          parallelContinuation0,
          parallelContinuation1
        )
    combined
  }

  def multiply[Buffer, Element, Operand0, Operand1, OutputLayer /* <: DeviceBufferLayer */ ](operand0: Operand0,
                                                                                             operand1: Operand1)(
      implicit
      deepLearning0: DeepLearning.Aux[Operand0, Buffer, Buffer],
      deepLearning1: DeepLearning.Aux[Operand1, Buffer, Buffer],
      isDoBuffer: Do[Tape[Buffer, Buffer]] <:< Do[Tape[DeviceBuffer[Element], DeviceBuffer[Element]]],
      layerFactory: ToLayer.Aux[DeviceBuffer[Element], DeviceBuffer[Element], OutputLayer],
      memory: Memory[Element]
  ): OutputLayer = {
    val operand0Forward: Do[Tape[DeviceBuffer[Element], DeviceBuffer[Element]]] = isDoBuffer(operand0.forward)
    val operand1Forward: Do[Tape[DeviceBuffer[Element], DeviceBuffer[Element]]] = isDoBuffer(operand1.forward)

    val Parallel(tupledDo) = Apply[ParallelDo].tuple2(Parallel(operand0Forward), Parallel(operand1Forward))

    val forward = tupledDo.flatMap {
      case (Tape(data0, backward0), Tape(data1, backward1)) =>
        val length = data0.length
        def outputData(data0: DeviceBuffer[Element], data1: DeviceBuffer[Element]): Do[DeviceBuffer[Element]] = {
          if (length != data1.length) {
            throw new IllegalArgumentException("The length of data0 should equal the length of data1")
          }

          val doOutputBuffer = allocateBuffer[Element](length).flatMap { output: DeviceBuffer[Element] =>
            Do.monadicCloseable(matrixMultiplyProgram.createFirstKernel())
              .flatMap { kernel =>
                kernel(0) = data0
                kernel(1) = data1
                kernel(2) = output
                val self: this.type = this
                val doEvent: Do[Event] = kernel.enqueue(length)(Witness(self))
                doEvent.flatMap { event =>
                  val doWait: Do[Unit] = Do.garbageCollected(event.waitForComplete())
                  doWait
                }
              }
              .intransitiveMap { _: Unit =>
                output
              }
          }
          doOutputBuffer
        }

        def backward(doOutputDelta: Do[DeviceBuffer[Element]]): UnitContinuation[Unit] = {
          val delta0: Do[DeviceBuffer[Element]] = doOutputDelta.flatMap { outputDelta: DeviceBuffer[Element] =>
            allocateBuffer[Element](outputDelta.length).flatMap { output: DeviceBuffer[Element] =>
              Do.monadicCloseable(multiplyProgram.createFirstKernel())
                .flatMap { kernel =>
                  kernel(0) = outputDelta
                  kernel(1) = data1
                  kernel(2) = output
                  val self: this.type = this
                  kernel.enqueue(length)(Witness(self)).flatMap { event =>
                    Do.garbageCollected(event.waitForComplete())
                  }
                }
                .intransitiveMap { _: Unit =>
                  output
                }
            }
          }
          val delta1: Do[DeviceBuffer[Element]] = doOutputDelta.flatMap { outputDelta: DeviceBuffer[Element] =>
            allocateBuffer[Element](outputDelta.length).flatMap { output: DeviceBuffer[Element] =>
              Do.monadicCloseable(multiplyProgram.createFirstKernel())
                .flatMap { kernel =>
                  kernel(0) = outputDelta
                  kernel(1) = data0
                  kernel(2) = output
                  val self: this.type = this
                  kernel.enqueue(length)(Witness(self)).flatMap { event =>
                    Do.garbageCollected(event.waitForComplete())
                  }
                }
                .intransitiveMap { _: Unit =>
                  output
                }
            }
          }

          parallelAppend(backward0(delta0), backward1(delta1))
        }
        outputData(data0, data1).map(Tape(_, backward))
    }
    layerFactory.toLayer(forward)
  }

  private lazy val subtractProgram: Program = {
    val program = createProgramWithSource(
      Seq(
        """
        kernel void subtract(global const float* restrict input0, global const float* restrict input1, global float* restrict output) {
          const size_t index = get_global_id(0);
          output[index] = input0[index] - input1[index];
        }
      """)
    )

    program.build()
    program
  }

  private lazy val negativeProgram: Program = {
    val program = createProgramWithSource(
      Seq("""
        kernel void negative(global const float* restrict input, global float* restrict output) {
          const size_t index = get_global_id(0);
          output[index] = -input[index];
        }
      """)
    )

    program.build()
    program
  }

  private lazy val multiplyProgram: Program = {
    val program = createProgramWithSource(
      Seq(
        """
        kernel void multiply(global const float* restrict input0, global const float* restrict input1, global float* restrict output) {
          const size_t index = get_global_id(0);
          output[index] = input0[index] * input1[index];
        }
      """)
    )

    program.build()
    program
  }

  private lazy val matrixMultiplyProgram: Program = {
    val program = createProgramWithSource(
      Seq(
        """
        kernel void matrix_multiply(global const float* restrict input0, global const float* restrict input1, global float* restrict output, size_t matrix0_columns, size_t matrix1_columns) {
          const size_t i = get_global_id(0);
          const size_t j = get_global_id(1);

          float value = 0.0f;
          for (int k = 0; k < matrix0_columns; ++k) {
            float elementA = input0[i * matrix0_columns + k];
            float elementB = input1[k * matrix1_columns + j];
            value += elementA * elementB;
          }
          output[i * matrix1_columns + j] = value;
        }
      """)
    )
    program.build()
    program
  }

  private lazy val backwardMatrixMultiplyProgram: Program = {
    val program = createProgramWithSource(
      Seq(
        """
        kernel void backward_matrix_multiply(global const float* restrict input0, global const float* restrict input1, global float* restrict output, size_t matrix0_columns, size_t matrix1_rows) {
          const size_t i = get_global_id(0);
          const size_t j = get_global_id(1);

          float value = 0.0f;
          for (int k = 0; k < matrix0_columns; ++k) {
            float elementA = input0[i * matrix0_columns + k];
            float elementB = input1[j * matrix0_columns + k];
            value += elementA * elementB;
          }
          output[i * matrix1_rows + j] = value;
        }
      """)
    )
    program.build()
    program
  }

}
