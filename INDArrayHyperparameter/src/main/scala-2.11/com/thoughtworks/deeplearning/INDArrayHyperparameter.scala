package com.thoughtworks.deeplearning

import java.util.logging.{Level, Logger}

import com.thoughtworks.deeplearning.logs.{UncaughtExceptionDuringBackward, WeightIsUpdating}
import com.thoughtworks.deeplearning.math._
import com.thoughtworks.deeplearning.math.polyFunctions
import com.thoughtworks.deeplearning.tapefactories.{MonoidOutput, SemigroupOutput, Unary}
import com.thoughtworks.deeplearning.Lift.LowPriorityLift
import com.thoughtworks.deeplearning._
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import com.thoughtworks.raii.covariant.{Releasable, ResourceT}
import org.nd4j.linalg.api.ops.impl.transforms.{IsMax, Sqrt}
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.ops.transforms.Transforms.{sign, sqrt}
import org.nd4j.linalg.util.ArrayUtil
import org.nd4s.Implicits._
import com.thoughtworks.each.Monadic._
import com.thoughtworks.raii.asynchronous
import shapeless.{HList, Lazy, Poly0, Witness, the}
import com.thoughtworks.deeplearning.math._
import com.thoughtworks.feature._
import com.thoughtworks.feature.byname.ByName
import com.thoughtworks.feature.Factory.inject

import scala.concurrent.ExecutionContext
import scala.util.{Failure, Success, Try}
import scalaz.{-\/, Monoid, Semigroup, \/, \/-}
import scalaz.concurrent.{Future, Task}
import scalaz.syntax.all._
import org.nd4j.linalg.api.ndarray.INDArray
import shapeless.PolyDefns.Case0
import shapeless.ops.hlist.Selector
import sourcecode.{FullName, Name}

import scala.annotation.meta.getter
import scala.concurrent.ExecutionContext
import scalaz.{\/, \/-}
import scalaz.concurrent.Task

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait INDArrayHyperparameter extends DoubleHyperparameter {

  protected implicit object INDArraySemigroup extends Semigroup[INDArray] {
    override def append(f1: INDArray, f2: => INDArray): INDArray = f1 + f2
  }

  private def jumpTask()(implicit executionContext: ExecutionContext): Task[Unit] = {
    Task.async { handler: ((Throwable \/ Unit) => Unit) =>
      executionContext.execute {
        new Runnable {
          override def run(): Unit = handler(\/-(()))
        }
      }
    }
  }

  // TODO: Add a test for this method and auto-broadcasting on n-dimension arrays for n > 2
  private def sumAs(outputDeltaValue: INDArray, shape: Array[Int]) = {
    val singleElementDimension = (shape: Seq[Int]).view.zip(outputDeltaValue.shape).zipWithIndex.collect {
      case ((1, originSize), dimension) if originSize > 1 => dimension
    }
    if (singleElementDimension.isEmpty) {
      outputDeltaValue
    } else {
      outputDeltaValue.sum(singleElementDimension.force: _*).reshape(shape: _*)
    }
  }

  private def autoBroadcastShape(shape1: Array[Int], shape2: Array[Int]) = {
    require(shape1.length == shape2.length)
    shape1.zip(shape2).map {
      case (1, bSize) => bSize
      case (aSize, 1) => aSize
      case (aSize, bSize) if aSize == bSize => aSize
    }
  }

  @inline
  protected def reciprocal[Operand](operand: Operand)(
      implicit liftOperand: Lift.Aux[Operand, INDArray, INDArray],
      fullName: sourcecode.FullName,
      caller: Caller[_],
      methodName: sourcecode.Name,
      executionContext: ExecutionContext): Do[Tape[INDArray, INDArray]] = {
    tapefactories.Unary.doTape(liftOperand(operand)) { data: INDArray =>
      throwableMonadic[Task] {
        val outputData = data rdiv 1.0
        val computeBackward = { outputDelta: INDArray =>
          throwableMonadic[Do] {
            Do.jump().each
            -outputDelta / (data * data)
          }
        }
        (outputData, computeBackward)
      }
    }
  }

  @inline
  def reshape[Operand](operand: Operand, newShape: Int*)(
      implicit liftOperand: Lift.Aux[Operand, INDArray, INDArray],
      fullName: sourcecode.FullName,
      caller: Caller[_],
      methodName: sourcecode.Name,
      executionContext: ExecutionContext): Do[Tape[INDArray, INDArray]] = {
    tapefactories.Unary.doTape(liftOperand(operand)) { (data: INDArray) =>
      throwableMonadic[Task] {
        jumpTask().each
        val dataShape = data.shape()
        val outputData = data.reshape(newShape: _*)
        val computeBackward = { outputDelta: INDArray =>
          throwableMonadic[Do] {
            Do.jump().each
            outputDelta.reshape(dataShape: _*)
          }
        }
        (outputData, computeBackward)
      }
    }
  }

  @inline
  def permute[Operand](operand: Operand, dimensions: Int*)(
      implicit liftOperand: Lift.Aux[Operand, INDArray, INDArray],
      fullName: sourcecode.FullName,
      caller: Caller[_],
      methodName: sourcecode.Name,
      executionContext: ExecutionContext): Do[Tape[INDArray, INDArray]] = {
    tapefactories.Unary.doTape(liftOperand(operand)) { (data: INDArray) =>
      throwableMonadic[Task] {
        jumpTask().each
        val dataShape = data.shape()
        val outputData = data.permute(dimensions: _*)
        val computeBackward = { outputDelta: INDArray =>
          throwableMonadic[Do] {
            Do.jump().each
            val indexedSeq: IndexedSeq[Int] = dataShape.indices
              .map { index =>
                dimensions.indexOf(index)
              }
            outputDelta.permute(indexedSeq: _*)
          }
        }
        (outputData, computeBackward)
      }
    }
  }

  @inline
  def sumT[Operand](operand: Operand)(implicit liftOperand: Lift.Aux[Operand, INDArray, INDArray],
                                      fullName: sourcecode.FullName,
                                      caller: Caller[_],
                                      methodName: sourcecode.Name,
                                      executionContext: ExecutionContext): Do[Tape[scala.Double, scala.Double]] = {
    tapefactories.Unary.doTape[INDArray, INDArray, scala.Double, scala.Double](liftOperand(operand)) {
      data: INDArray =>
        throwableMonadic[Task] {
          jumpTask().each
          val outputData = data.sumT
          val computeBackward = { outputDelta: scala.Double =>
            throwableMonadic[Do] {
              Do.jump().each
              Nd4j.valueArrayOf(data.shape(), outputDelta)
            }
          }
          (outputData, computeBackward)
        }
    }
  }

  @inline
  def sum[Operand](operand: Operand, dimensions: Int*)(
      implicit liftOperand: Lift.Aux[Operand, INDArray, INDArray],
      fullName: sourcecode.FullName,
      caller: Caller[_],
      methodName: sourcecode.Name,
      executionContext: ExecutionContext): Do[Tape[INDArray, INDArray]] = {
    tapefactories.Unary.doTape(liftOperand(operand)) { data: INDArray =>
      throwableMonadic[Task] {
        jumpTask().each
        val outputData = data.sum(dimensions: _*)
        val computeBackward = { outputDelta: INDArray =>
          throwableMonadic[Do] {
            Do.jump().each
            outputDelta.broadcast(data.shape(): _*)
          }
        }
        (outputData, computeBackward)
      }
    }
  }

  implicit final class DifferentiableINDArrayOps[Operand](operand: Operand)(
      implicit liftOperand: Lift.Aux[Operand, INDArray, INDArray],
      fullName: sourcecode.FullName,
      methodName: sourcecode.Name,
      caller: Caller[_],
      executionContext: ExecutionContext) {
    @inline
    def unary_- : Do[Tape[INDArray, INDArray]] = {
      tapefactories.Unary.doTape(liftOperand(operand)) { data =>
        Task.delay {
          val outputData = -data
          val computeBackward = { outputDelta: INDArray =>
            throwableMonadic[Do] {
              Do.jump().each
              -outputDelta
            }
          }
          (outputData, computeBackward)
        }
      }
    }
  }

  @inline
  def conv2d[Input, Weight, Bias](input: Input,
                                  weight: Weight,
                                  bias: Bias,
                                  kernel: (Int, Int),
                                  stride: (Int, Int),
                                  padding: (Int, Int))(
      implicit liftInput: Lift.Aux[Input, INDArray, INDArray],
      liftWeight: Lift.Aux[Weight, INDArray, INDArray],
      liftBias: Lift.Aux[Bias, INDArray, INDArray],
      fullName: sourcecode.FullName,
      caller: Caller[_],
      methodName: sourcecode.Name,
      executionContext: ExecutionContext): Do[Tape[INDArray, INDArray]] = {
    def monadicConv2d(input: Do[Tape[INDArray, INDArray]],
                      weight: Do[Tape[INDArray, INDArray]],
                      bias: Do[Tape[INDArray, INDArray]]) = monadic[Do] {
      val inputShape: Array[Int] = input.each.data.shape()
      val count = inputShape(0)
      val depth = inputShape(1)
      val yAxis = inputShape(2)
      val xAxis = inputShape(3)

      val weightShape: Array[Int] = weight.each.data.shape()

      val kernelNumber = weightShape(0)
      val col = im2col(input, kernel, stride, padding)
      val permutedCol: Do[Tape[INDArray, INDArray]] = permute(col, 0, 4, 5, 1, 2, 3)
      val depthKernelKernel = depth * kernel._1 * kernel._2
      val countXAisYAis = count * yAxis * xAxis

      val operandCol2d = reshape(permutedCol, countXAisYAis, depthKernelKernel)

      val reshapedWeight = reshape(weight, kernelNumber, depthKernelKernel)

      val permutedWeight = permute(reshapedWeight, 1, 0)

      val dotResult: Do[Tape[INDArray, INDArray]] = dot(operandCol2d, permutedWeight)

      val plusResult: Do[Tape[INDArray, INDArray]] =
        implicits.`INDArray+INDArray`.apply(dotResult, bias: Do[Tape[INDArray, INDArray]])

      val reshapeResult = reshape(plusResult, count, yAxis, xAxis, kernelNumber)

      permute(reshapeResult, 0, 3, 1, 2).each

    }

    monadicConv2d(liftInput(input), liftWeight(weight), liftBias(bias))
  }

  @inline
  def dot[Left, Right](left: Left, right: Right)(implicit liftLeft: Lift.Aux[Left, INDArray, INDArray],
                                                 liftRight: Lift.Aux[Right, INDArray, INDArray],
                                                 fullName: sourcecode.FullName,
                                                 caller: Caller[_],
                                                 methodName: sourcecode.Name,
                                                 executionContext: ExecutionContext): Do[Tape[INDArray, INDArray]] = {
    tapefactories.Binary.doTape(liftLeft(left), liftRight(right)) { (data0: INDArray, data1: INDArray) =>
      throwableMonadic[Task] {
        jumpTask().each

        val outputData = data0 dot data1

        val computeBackward = { outputDelta: INDArray =>
          val delta0Future =
            throwableMonadic[Do] {
              Do.jump().each
              outputDelta dot data1.T
            }

          val delta1Future =
            throwableMonadic[Do] {
              Do.jump().each
              data0.T dot outputDelta
            }
          (delta0Future, delta1Future)
        }
        (outputData, computeBackward)
      }
    }
  }

  private def toArray(tuple2: (Int, Int)): Array[Int] = {
    val (one, two) = tuple2
    Array(one, two)
  }

  @inline
  def im2col[Operand](operand: Operand, kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int))(
      implicit liftOperand: Lift.Aux[Operand, INDArray, INDArray],
      fullName: sourcecode.FullName,
      caller: Caller[_],
      methodName: sourcecode.Name,
      executionContext: ExecutionContext): Do[Tape[INDArray, INDArray]] = {
    tapefactories.Unary.doTape(liftOperand(operand)) { data: INDArray =>
      throwableMonadic[Task] {
        jumpTask().each
        val dataShape = data.shape()
        val strideArray = toArray(stride)
        val paddingArray = toArray(padding)
        val outputData = Convolution.im2col(data, toArray(kernel), strideArray, paddingArray)
        val computeBackward = { outputDelta: INDArray =>
          throwableMonadic[Do] {
            Do.jump().each
            Convolution.col2im(outputDelta, strideArray, paddingArray, dataShape(2), dataShape(3))
          }
        }
        (outputData, computeBackward)
      }
    }
  }

  @inline
  def mean[Operand](operand: Operand)(implicit liftOperand: Lift.Aux[Operand, INDArray, INDArray],
                                      fullName: sourcecode.FullName,
                                      caller: Caller[_],
                                      methodName: sourcecode.Name,
                                      executionContext: ExecutionContext): Do[Tape[scala.Double, scala.Double]] = {
    tapefactories.Unary.doTape(liftOperand(operand)) { data: INDArray =>
      throwableMonadic[Task] {
        jumpTask().each
        val outputData = data.sumT / ArrayUtil.prod(data.shape(): _*)
        val computeBackward = { outputDelta: scala.Double =>
          throwableMonadic[Do] {
            Do.jump().each
            Nd4j.valueArrayOf(data.shape(), outputDelta)
          }
        }
        (outputData, computeBackward)
      }
    }
  }
  trait INDArrayWeightApi { this: INDArrayWeight =>
    implicit val fullName: sourcecode.FullName
    implicit val name: sourcecode.Name
    implicit val caller: Caller[_]
    implicit val executionContext: ExecutionContext

    var data: INDArray

    private def optimizer[SubtypeOfOptimizer](delta: INDArray)(
        implicit implicitApplyRest: ImplicitApply.Aux[indArrayPartialApplyOriginalDelta.Rest, SubtypeOfOptimizer],
        constraint: SubtypeOfOptimizer <:< INDArrayOptimizer
    ): INDArrayOptimizer = {
      constraint(
        implicitApplyRest(
          indArrayPartialApplyOriginalDelta(indArraypartialApplyWeight(indArrayOptimizerFactory.newInstance,
                                                                       indArrayWeightParameter(this)),
                                            indArrayOriginalDeltaParameter(delta))))
    }

    private[INDArrayHyperparameter] final def backward[SubtypeOfOptimizer](deltaFuture: Do[INDArray])(
        implicit implicitApplyRest: ImplicitApply.Aux[indArrayPartialApplyOriginalDelta.Rest, SubtypeOfOptimizer],
        constraint: SubtypeOfOptimizer <:< INDArrayOptimizer): Future[Unit] = {

      Do.run(Do.releaseFlatMap(deltaFuture) { delta =>
          Do.jump().map { unit: Unit =>
            data -= optimizer(delta).delta
            ()
          }
        })
        .get
        .map {
          case \/-(()) => ()
          case -\/(e) =>
            val logRecord = UncaughtExceptionDuringBackward(e)
            logger.log(logRecord)
        }

    }

  }

  type INDArrayWeight <: INDArrayWeightApi

  object INDArrayWeight {
    def apply[SubtypeOfWeight, OptimizerFunction, Optimizer](data: INDArray)(
        implicit implicitApplyRest: ImplicitApply.Aux[indArrayPartialApplyData.Rest, SubtypeOfWeight],
        asINDArrayWeight: SubtypeOfWeight <:< INDArrayWeight
    ): INDArrayWeight = {
      asINDArrayWeight(
        implicitApplyRest(indArrayPartialApplyData(indArrayWeightFactory.newInstance, indArrayDataParameter(data))))
    }
  }
  trait INDArrayOptimizerApi {

    val weight: INDArrayWeight

    val originalDelta: INDArray
    def delta: INDArray = originalDelta
  }
  type INDArrayOptimizer <: INDArrayOptimizerApi

  @(inject @getter)
  protected val indArrayOptimizerFactory: Factory[INDArrayOptimizer]

  @(inject @getter)
  protected val indArraypartialApplyWeight: PartialApply[indArrayOptimizerFactory.Constructor, Witness.`"weight"`.T]

  @inject
  protected def indArrayWeightParameter: INDArrayWeight <:< indArraypartialApplyWeight.Parameter

  @(inject @getter)
  protected val indArrayPartialApplyOriginalDelta: PartialApply[indArraypartialApplyWeight.Rest,
                                                                Witness.`"originalDelta"`.T]

  @inject
  protected def indArrayOriginalDeltaParameter: INDArray <:< indArrayPartialApplyOriginalDelta.Parameter

  @(inject @getter)
  protected val indArrayWeightFactory: Factory[INDArrayWeight]

  @(inject @getter)
  protected val indArrayPartialApplyData: PartialApply[indArrayWeightFactory.Constructor, Witness.`"data"`.T]

  @inject
  protected def indArrayDataParameter: INDArray <:< indArrayPartialApplyData.Parameter

  trait ImplicitsApi extends super.ImplicitsApi {

    @inline
    implicit def liftINDArray[A](
        implicit typeClass: LowPriorityLift.Aux[A, INDArray, INDArray]): Lift.Aux[A, INDArray, INDArray] =
      typeClass

    @inline
    implicit def liftINDArrayWeight[SubtypeOfOptimizer](
        implicit implicitApplyRest: ImplicitApply.Aux[indArrayPartialApplyOriginalDelta.Rest, SubtypeOfOptimizer],
        constraint: SubtypeOfOptimizer <:< INDArrayOptimizer): Lift.Aux[INDArrayWeight, INDArray, INDArray] = {
      new Lift[INDArrayWeight] {
        override type Data = INDArray
        override type Delta = INDArray

        @inline
        override def apply(weight: INDArrayWeight): Do[Tape[INDArray, INDArray]] = {
          import weight._
          Do.now(Tape(data, backward[SubtypeOfOptimizer]))
        }
      }
    }

    implicit object INDArrayLoss extends Loss[INDArray, INDArray] {
      override def deltaLoss(data: INDArray): Do[INDArray] = Do.now(Nd4j.ones(data.shape(): _*))
    }

    @inline
    implicit def `INDArray+INDArray`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions.+.Case.Aux[Do[Tape[INDArray, INDArray]],
                                                                      Do[Tape[INDArray, INDArray]],
                                                                      Do[Tape[INDArray, INDArray]]] = {
      polyFunctions.+.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0: INDArray, data1: INDArray) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = {
              val newShape = autoBroadcastShape(data0.shape(), data1.shape())
              data0.broadcast(newShape: _*) + data1.broadcast(newShape: _*)
            }
            val computeBackward = { outputDelta: INDArray =>
              val delta0Future = throwableMonadic[Do] {
                Do.jump().each
                sumAs(outputDelta, data0.shape())
              }
              val delta1Future = throwableMonadic[Do] {
                Do.jump().each
                sumAs(outputDelta, data1.shape())
              }
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `INDArray+Double`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions.+.Case.Aux[Do[Tape[INDArray, INDArray]],
                                                                      Do[Tape[scala.Double, scala.Double]],
                                                                      Do[Tape[INDArray, INDArray]]] = {
      polyFunctions.+.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0: INDArray, data1: scala.Double) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = data0 + data1
            val computeBackward = { outputDelta: INDArray =>
              val delta0Future = Do.now(outputDelta)
              val delta1Future = throwableMonadic[Do] {
                Do.jump().each
                outputDelta.sumT
              }
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `Double+INDArray`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions.+.Case.Aux[Do[Tape[scala.Double, scala.Double]],
                                                                      Do[Tape[INDArray, INDArray]],
                                                                      Do[Tape[INDArray, INDArray]]] = {
      polyFunctions.+.at { (operand0, operand1) =>
        import math.PolyOps
        operand1.+(operand0)(Lift.LiftCase2.liftCase2(Lift.fromSubtype, Lift.fromSubtype, `INDArray+Double`))
      }
    }

    @inline
    implicit def `INDArray-INDArray`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions.-.Case.Aux[Do[Tape[INDArray, INDArray]],
                                                                      Do[Tape[INDArray, INDArray]],
                                                                      Do[Tape[INDArray, INDArray]]] = {
      polyFunctions.-.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0: INDArray, data1: INDArray) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = {
              val newShape = autoBroadcastShape(data0.shape(), data1.shape())
              data0.broadcast(newShape: _*) - data1.broadcast(newShape: _*)
            }
            val computeBackward = { outputDelta: INDArray =>
              val delta0Future = throwableMonadic[Do] {
                Do.jump().each
                sumAs(outputDelta, data0.shape())
              }
              val delta1Future = throwableMonadic[Do] {
                Do.jump().each
                sumAs(-outputDelta, data1.shape())
              }
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `INDArray-Double`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions.-.Case.Aux[Do[Tape[INDArray, INDArray]],
                                                                      Do[Tape[scala.Double, scala.Double]],
                                                                      Do[Tape[INDArray, INDArray]]] = {
      polyFunctions.-.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0: INDArray, data1: scala.Double) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = data0 - data1
            val computeBackward = { outputDelta: INDArray =>
              val delta0Future = Do.now(outputDelta)
              val delta1Future = throwableMonadic[Do] {
                Do.jump().each
                -outputDelta.sumT
              }
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `Double-INDArray`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions.-.Case.Aux[Do[Tape[scala.Double, scala.Double]],
                                                                      Do[Tape[INDArray, INDArray]],
                                                                      Do[Tape[INDArray, INDArray]]] = {
      polyFunctions.-.at { (operand0, operand1) =>
        import shapeless.syntax.singleton._
        `INDArray+Double`.apply(new DifferentiableINDArrayOps(operand1).unary_-, operand0)
      }
    }

    @inline
    implicit def `INDArray*INDArray`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions.*.Case.Aux[Do[Tape[INDArray, INDArray]],
                                                                      Do[Tape[INDArray, INDArray]],
                                                                      Do[Tape[INDArray, INDArray]]] = {
      polyFunctions.*.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0: INDArray, data1: INDArray) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = {
              val newShape = autoBroadcastShape(data0.shape(), data1.shape())
              data0.broadcast(newShape: _*) * data1.broadcast(newShape: _*)
            }
            val computeBackward = { outputDelta: INDArray =>
              val delta0Future = throwableMonadic[Do] {
                Do.jump().each
                sumAs(data1.broadcast(outputDelta.shape(): _*) * outputDelta, data0.shape())
              }
              val delta1Future = throwableMonadic[Do] {
                Do.jump().each
                sumAs(data0.broadcast(outputDelta.shape(): _*) * outputDelta, data1.shape())
              }
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `INDArray*Double`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions.*.Case.Aux[Do[Tape[INDArray, INDArray]],
                                                                      Do[Tape[scala.Double, scala.Double]],
                                                                      Do[Tape[INDArray, INDArray]]] = {
      polyFunctions.*.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0: INDArray, data1: scala.Double) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = data0 * data1
            val computeBackward = { outputDelta: INDArray =>
              val delta0Future = throwableMonadic[Do] {
                Do.jump().each
                outputDelta * data1
              }
              val delta1Future = throwableMonadic[Do] {
                (data0 * outputDelta).sumT
              }
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `Double*INDArray`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions.*.Case.Aux[Do[Tape[scala.Double, scala.Double]],
                                                                      Do[Tape[INDArray, INDArray]],
                                                                      Do[Tape[INDArray, INDArray]]] = {
      polyFunctions.*.at { (operand0, operand1) =>
        `INDArray*Double`.apply(operand1, operand0)

      }
    }

    @inline
    implicit def `INDArray/INDArray`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions./.Case.Aux[Do[Tape[INDArray, INDArray]],
                                                                      Do[Tape[INDArray, INDArray]],
                                                                      Do[Tape[INDArray, INDArray]]] = {
      polyFunctions./.at { (operand0, operand1) =>
        operand0 * reciprocal(operand1)
      }
    }

    @inline
    implicit def `INDArray/Double`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions./.Case.Aux[Do[Tape[INDArray, INDArray]],
                                                                      Do[Tape[scala.Double, scala.Double]],
                                                                      Do[Tape[INDArray, INDArray]]] = {
      polyFunctions./.at { (operand0, operand1) =>
        `INDArray*Double`.apply(operand0, doubleReciprocal(operand1))
      }
    }

    @inline
    implicit def `Double/INDArray`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions./.Case.Aux[Do[Tape[scala.Double, scala.Double]],
                                                                      Do[Tape[INDArray, INDArray]],
                                                                      Do[Tape[INDArray, INDArray]]] = {
      polyFunctions./.at { (operand0, operand1) =>
        operand0 * reciprocal(operand1)
      }
    }

    @inline
    implicit def `max(INDArray,Double)`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions.max.Case.Aux[Do[Tape[INDArray, INDArray]],
                                                                        Do[Tape[scala.Double, scala.Double]],
                                                                        Do[Tape[INDArray, INDArray]]] = {
      polyFunctions.max.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0: INDArray, data1: scala.Double) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = Transforms.max(data0, data1)
            val computeBackward = { outputDelta: INDArray =>
              val delta0Future = throwableMonadic[Do] {
                Do.jump().each
                (data0 gt data1) * outputDelta
              }
              val delta1Future = throwableMonadic[Do] {
                Do.jump().each
                ((data0 lt data1) * outputDelta).sumT
              }
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `min(INDArray,Double)`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions.min.Case.Aux[Do[Tape[INDArray, INDArray]],
                                                                        Do[Tape[scala.Double, scala.Double]],
                                                                        Do[Tape[INDArray, INDArray]]] = {
      polyFunctions.min.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0: INDArray, data1: scala.Double) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = Transforms.min(data0, data1)
            val computeBackward = { outputDelta: INDArray =>
              val delta0Future = throwableMonadic[Do] {
                Do.jump().each
                (data0 lt data1) * outputDelta
              }
              val delta1Future = throwableMonadic[Do] {
                Do.jump().each
                ((data0 gt data1) * outputDelta).sumT
              }
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `exp(INDArray)`(implicit fullName: sourcecode.FullName,
                                 caller: Caller[_],
                                 methodName: sourcecode.Name,
                                 executionContext: ExecutionContext)
      : polyFunctions.exp.Case.Aux[Do[Tape[INDArray, INDArray]], Do[Tape[INDArray, INDArray]]] = {
      polyFunctions.exp.at { operand =>
        tapefactories.Unary.doTape(operand) { (data: INDArray) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = Transforms.exp(data)
            val computeBackward = { outputDelta: INDArray =>
              throwableMonadic[Do] {
                Do.jump().each
                outputData * outputDelta
              }
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `log(INDArray)`(implicit fullName: sourcecode.FullName,
                                 caller: Caller[_],
                                 methodName: sourcecode.Name,
                                 executionContext: ExecutionContext)
      : polyFunctions.log.Case.Aux[Do[Tape[INDArray, INDArray]], Do[Tape[INDArray, INDArray]]] = {
      polyFunctions.log.at { operand =>
        tapefactories.Unary.doTape(operand) { (data: INDArray) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = Transforms.log(data)
            val computeBackward = { outputDelta: INDArray =>
              throwableMonadic[Do] {
                Do.jump().each
                outputDelta / data
              }
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `abs(INDArray)`(implicit fullName: sourcecode.FullName,
                                 caller: Caller[_],
                                 methodName: sourcecode.Name,
                                 executionContext: ExecutionContext)
      : polyFunctions.abs.Case.Aux[Do[Tape[INDArray, INDArray]], Do[Tape[INDArray, INDArray]]] = {
      polyFunctions.abs.at { operand =>
        tapefactories.Unary.doTape(operand) { (data: INDArray) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = Transforms.abs(data)
            val computeBackward = { outputDelta: INDArray =>
              throwableMonadic[Do] {
                Do.jump().each
                outputDelta * Transforms.sign(data)
              }
            }
            (outputData, computeBackward)
          }
        }
      }
    }
  }
  override type Implicits <: ImplicitsApi
  type INDArrayLayer = Do[Tape[scala.Float, scala.Float]]

}
