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

  // Workaround for https://github.com/milessabin/shapeless/issues/755
  implicit private def witnessThis: Witness.Aux[this.type] = Witness.mkWitness(this)

  trait ImplicitsApi extends super[Layers].ImplicitsApi with super[OpenCL].ImplicitsApi

  override type Implicits <: ImplicitsApi

  trait DeviceBufferLayerApi extends LayerApi {

    override type Data = DeviceBuffer[Element]
    override type Delta = DeviceBuffer[Element]
    type Element
  }

  type DeviceBufferLayer <: DeviceBufferLayerApi with Layer

  /** @note Both `operand0` and `operand1` are differentiable row-major order matrix.
    *       The first dimension is its height, and the second dimension is its width.
    */
  def matrixMultiply[Operand0, Operand1, Buffer, Element, OutputLayer /* <: DeviceBufferLayer */ ](operand0: Operand0,
                                                                                                   operand1: Operand1,
                                                                                                   width0: Int)(
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
        val length0 = data0.length
        val lengh1 = data1.length
        if (length0 % width0 != 0) {
          throw new IllegalArgumentException("The data0 should be a matrix")
        }

        val height0 = length0 / width0
        if (lengh1 % width0 != 0) {
          throw new IllegalArgumentException("The data1 should be a matrix")
        }

        val width1 = lengh1 / width0
//        println(s"[$height0, $width0] x [$width0, $width1] = [$height0, $width1]")
        def outputData(data0: DeviceBuffer[Element], data1: DeviceBuffer[Element]): Do[DeviceBuffer[Element]] = {
          val doOutputBuffer = allocateBuffer[Element](height0 * width1).flatMap { output: DeviceBuffer[Element] =>
            Do.monadicCloseable(matrixMultiplyForwardProgram.createFirstKernel())
              .flatMap { kernel =>
                kernel(0) = data0
                kernel(1) = data1
                kernel(2) = output
                kernel(3) = width0
                val self: this.type = this
                val doEvent: Do[Event] = kernel.enqueue(height0, width1)(Witness(self))
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
            if (outputDelta.length != height0 * width1) {
              throw new IllegalArgumentException("The outputDelta should be a matrix")
            }

            allocateBuffer[Element](height0 * width0).flatMap { delta0: DeviceBuffer[Element] =>
              Do.monadicCloseable(matrixMultiplyBackwardDelta0Program.createFirstKernel())
                .flatMap { kernel =>
                  kernel(0) = outputDelta
                  kernel(1) = data1
                  kernel(2) = delta0
                  kernel(3) = width1
                  val self: this.type = this
                  kernel.enqueue(height0, width0)(Witness(self)).flatMap { event =>
                    Do.garbageCollected(event.waitForComplete())
                  }
                }
                .intransitiveMap { _: Unit =>
                  delta0
                }
            }
          }
//          val delta1: Do[DeviceBuffer[Element]] = ???
          val delta1: Do[DeviceBuffer[Element]] = doOutputDelta.flatMap { outputDelta: DeviceBuffer[Element] =>
            allocateBuffer[Element](width0 * width1).flatMap { delta1: DeviceBuffer[Element] =>
              Do.monadicCloseable(matrixMultiplyBackwardDelta1Program.createFirstKernel())
                .flatMap { kernel =>
                  kernel(0) = outputDelta
                  kernel(1) = data0
                  kernel(2) = delta1
                  kernel(3) = height0
                  val self: this.type = this
                  kernel.enqueue(width0, width1)(Witness(self)).flatMap { event =>
                    Do.garbageCollected(event.waitForComplete())
                  }
                }
                .intransitiveMap { _: Unit =>
                  delta1

                }
            }

          }
          parallelAppend(backward0(delta0), backward1(delta1))
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
            Do.monadicCloseable(multiplyProgram.createFirstKernel())
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

  private lazy val matrixMultiplyForwardProgram: Program = {
    val program = createProgramWithSource(
      Seq("""
        kernel void matrix_multiply_forward(global const float* const restrict data0,
                                    global const float* const restrict data1,
                                    global float* const restrict output,
                                    const uint width0) {
          const size_t y0 = get_global_id(0);

          const size_t width1 = get_global_size(1);
          const size_t x1 = get_global_id(1);

          float accumulator = 0.0f;
          for (size_t x0 = 0; x0 < width0; ++x0) {
            accumulator += (data0[y0 * width0 + x0] * data1[x0 * width1 + x1]);
          }
          output[y0 * width1 + x1] = accumulator;
        }
      """)
    )
    program.build()
    program
  }

  private lazy val matrixMultiplyBackwardDelta0Program: Program = {
    val program = createProgramWithSource(
      Seq("""
        kernel void matrix_multiply_backward_delta0(global const float* const restrict output_delta,
                                                    global const float* const restrict data1,
                                                    global float* const restrict delta0,
                                                    const uint width1) {
          const size_t y0 = get_global_id(0);
          const size_t width0 = get_global_size(1);
          const size_t x0 = get_global_id(1);

          float accumulator = 0.0f;
          for (size_t x1 = 0; x1 < width1; ++x1) {
            accumulator += (output_delta[y0 * width1 + x1] * data1[x0 * width1 + x1]);
          }
          delta0[y0 * width0 + x0] = accumulator;
        }
      """)
    )
    program.build()
    program
  }

  private lazy val matrixMultiplyBackwardDelta1Program: Program = {
    val program = createProgramWithSource(
      Seq("""
        kernel void matrix_multiply_backward_delta1(global const float* const restrict output_delta,
                                                    global const float* const restrict data0,
                                                    global float* const restrict delta1,
                                                    const uint height0) {
          const size_t width0 = get_global_size(0);
          const size_t x0 = get_global_id(0);
          const size_t width1 = get_global_size(1);
          const size_t x1 = get_global_id(1);

          float accumulator = 0.0f;
          for (int y0 = 0; y0 < height0; ++y0) {
            accumulator += (output_delta[y0 * width1 + x1] * data0[y0 * width0 + x0]);
          }
          delta1[x0 * width1 + x1] = accumulator;
        }
      """)
    )

    program.build()
    program
  }

}
