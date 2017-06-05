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
import org.nd4j.linalg.api.ndarray.{INDArray => Nd4jArray}
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

  protected implicit object INDArraySemigroup extends Semigroup[Nd4jArray] {
    override def append(f1: Nd4jArray, f2: => Nd4jArray): Nd4jArray = f1 + f2
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
  private def sumAs(outputDeltaValue: Nd4jArray, shape: Array[Int]) = {
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
      implicit liftOperand: Lift.Aux[Operand, Nd4jArray, Nd4jArray],
      fullName: sourcecode.FullName,
      caller: Caller[_],
      methodName: sourcecode.Name,
      executionContext: ExecutionContext): Do[Tape[Nd4jArray, Nd4jArray]] = {
    tapefactories.Unary.doTape(liftOperand(operand)) { data: Nd4jArray =>
      throwableMonadic[Task] {
        val outputData = data rdiv 1.0
        val computeBackward = { outputDelta: Nd4jArray =>
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
      implicit liftOperand: Lift.Aux[Operand, Nd4jArray, Nd4jArray],
      fullName: sourcecode.FullName,
      caller: Caller[_],
      methodName: sourcecode.Name,
      executionContext: ExecutionContext): Do[Tape[Nd4jArray, Nd4jArray]] = {
    tapefactories.Unary.doTape(liftOperand(operand)) { (data: Nd4jArray) =>
      throwableMonadic[Task] {
        jumpTask().each
        val dataShape = data.shape()
        val outputData = data.reshape(newShape: _*)
        val computeBackward = { outputDelta: Nd4jArray =>
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
      implicit liftOperand: Lift.Aux[Operand, Nd4jArray, Nd4jArray],
      fullName: sourcecode.FullName,
      caller: Caller[_],
      methodName: sourcecode.Name,
      executionContext: ExecutionContext): Do[Tape[Nd4jArray, Nd4jArray]] = {
    tapefactories.Unary.doTape(liftOperand(operand)) { (data: Nd4jArray) =>
      throwableMonadic[Task] {
        jumpTask().each
        val dataShape = data.shape()
        val outputData = data.permute(dimensions: _*)
        val computeBackward = { outputDelta: Nd4jArray =>
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
  def sumT[Operand](operand: Operand)(implicit liftOperand: Lift.Aux[Operand, Nd4jArray, Nd4jArray],
                                      fullName: sourcecode.FullName,
                                      caller: Caller[_],
                                      methodName: sourcecode.Name,
                                      executionContext: ExecutionContext): Do[Tape[scala.Double, scala.Double]] = {
    tapefactories.Unary.doTape[Nd4jArray, Nd4jArray, scala.Double, scala.Double](liftOperand(operand)) {
      data: Nd4jArray =>
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
      implicit liftOperand: Lift.Aux[Operand, Nd4jArray, Nd4jArray],
      fullName: sourcecode.FullName,
      caller: Caller[_],
      methodName: sourcecode.Name,
      executionContext: ExecutionContext): Do[Tape[Nd4jArray, Nd4jArray]] = {
    tapefactories.Unary.doTape(liftOperand(operand)) { data: Nd4jArray =>
      throwableMonadic[Task] {
        jumpTask().each
        val outputData = data.sum(dimensions: _*)
        val computeBackward = { outputDelta: Nd4jArray =>
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
      implicit liftOperand: Lift.Aux[Operand, Nd4jArray, Nd4jArray],
      fullName: sourcecode.FullName,
      methodName: sourcecode.Name,
      caller: Caller[_],
      executionContext: ExecutionContext) {
    @inline
    def unary_- : Do[Tape[Nd4jArray, Nd4jArray]] = {
      tapefactories.Unary.doTape(liftOperand(operand)) { data =>
        Task.delay {
          val outputData = -data
          val computeBackward = { outputDelta: Nd4jArray =>
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
      implicit liftInput: Lift.Aux[Input, Nd4jArray, Nd4jArray],
      liftWeight: Lift.Aux[Weight, Nd4jArray, Nd4jArray],
      liftBias: Lift.Aux[Bias, Nd4jArray, Nd4jArray],
      fullName: sourcecode.FullName,
      caller: Caller[_],
      methodName: sourcecode.Name,
      executionContext: ExecutionContext): Do[Tape[Nd4jArray, Nd4jArray]] = {
    def monadicConv2d(input: Do[Tape[Nd4jArray, Nd4jArray]],
                      weight: Do[Tape[Nd4jArray, Nd4jArray]],
                      bias: Do[Tape[Nd4jArray, Nd4jArray]]) = monadic[Do] {
      val inputShape: Array[Int] = input.each.data.shape()
      val count = inputShape(0)
      val depth = inputShape(1)
      val yAxis = inputShape(2)
      val xAxis = inputShape(3)

      val weightShape: Array[Int] = weight.each.data.shape()

      val kernelNumber = weightShape(0)
      val col = im2col(input, kernel, stride, padding)
      val permutedCol: Do[Tape[Nd4jArray, Nd4jArray]] = permute(col, 0, 4, 5, 1, 2, 3)
      val depthKernelKernel = depth * kernel._1 * kernel._2
      val countXAisYAis = count * yAxis * xAxis

      val operandCol2d = reshape(permutedCol, countXAisYAis, depthKernelKernel)

      val reshapedWeight = reshape(weight, kernelNumber, depthKernelKernel)

      val permutedWeight = permute(reshapedWeight, 1, 0)

      val dotResult: Do[Tape[Nd4jArray, Nd4jArray]] = dot(operandCol2d, permutedWeight)

      val plusResult: Do[Tape[Nd4jArray, Nd4jArray]] =
        implicits.`INDArray+INDArray`.apply(dotResult, bias: Do[Tape[Nd4jArray, Nd4jArray]])

      val reshapeResult = reshape(plusResult, count, yAxis, xAxis, kernelNumber)

      permute(reshapeResult, 0, 3, 1, 2).each

    }

    monadicConv2d(liftInput(input), liftWeight(weight), liftBias(bias))
  }

  @inline
  def dot[Left, Right](left: Left, right: Right)(
      implicit liftLeft: Lift.Aux[Left, Nd4jArray, Nd4jArray],
      liftRight: Lift.Aux[Right, Nd4jArray, Nd4jArray],
      fullName: sourcecode.FullName,
      caller: Caller[_],
      methodName: sourcecode.Name,
      executionContext: ExecutionContext): Do[Tape[Nd4jArray, Nd4jArray]] = {
    tapefactories.Binary.doTape(liftLeft(left), liftRight(right)) { (data0: Nd4jArray, data1: Nd4jArray) =>
      throwableMonadic[Task] {
        jumpTask().each

        val outputData = data0 dot data1

        val computeBackward = { outputDelta: Nd4jArray =>
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
      implicit liftOperand: Lift.Aux[Operand, Nd4jArray, Nd4jArray],
      fullName: sourcecode.FullName,
      caller: Caller[_],
      methodName: sourcecode.Name,
      executionContext: ExecutionContext): Do[Tape[Nd4jArray, Nd4jArray]] = {
    tapefactories.Unary.doTape(liftOperand(operand)) { data: Nd4jArray =>
      throwableMonadic[Task] {
        jumpTask().each
        val dataShape = data.shape()
        val strideArray = toArray(stride)
        val paddingArray = toArray(padding)
        val outputData = Convolution.im2col(data, toArray(kernel), strideArray, paddingArray)
        val computeBackward = { outputDelta: Nd4jArray =>
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
  def mean[Operand](operand: Operand)(implicit liftOperand: Lift.Aux[Operand, Nd4jArray, Nd4jArray],
                                      fullName: sourcecode.FullName,
                                      caller: Caller[_],
                                      methodName: sourcecode.Name,
                                      executionContext: ExecutionContext): Do[Tape[scala.Double, scala.Double]] = {
    tapefactories.Unary.doTape(liftOperand(operand)) { data: Nd4jArray =>
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

    var data: Nd4jArray

    private def optimizer[SubtypeOfOptimizer](delta: Nd4jArray)(
        implicit implicitApplyRest: ImplicitApply.Aux[partialApplyOriginalDelta.Rest, SubtypeOfOptimizer],
        constraint: SubtypeOfOptimizer <:< INDArrayOptimizer
    ): INDArrayOptimizer = {
      constraint(
        implicitApplyRest(
          partialApplyOriginalDelta(partialApplyWeight(optimizerFactory.newInstance, weightParameter(this)),
                                    originalDeltaParameter(delta))))
    }

    final def forward[SubtypeOfOptimizer](
        implicit implicitApplyRest: ImplicitApply.Aux[partialApplyOriginalDelta.Rest, SubtypeOfOptimizer],
        constraint: SubtypeOfOptimizer <:< INDArrayOptimizer): Do[Tape[Nd4jArray, Nd4jArray]] = {
      Do.now(Tape(data, backward[SubtypeOfOptimizer]))
    }

    private[INDArrayHyperparameter] final def backward[SubtypeOfOptimizer](deltaFuture: Do[Nd4jArray])(
        implicit implicitApplyRest: ImplicitApply.Aux[partialApplyOriginalDelta.Rest, SubtypeOfOptimizer],
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
    def apply[SubtypeOfWeight, OptimizerFunction, Optimizer](data: Nd4jArray)(
        implicit implicitApplyRest: ImplicitApply.Aux[partialApplyData.Rest, SubtypeOfWeight],
        asINDArrayWeight: SubtypeOfWeight <:< INDArrayWeight
    ): INDArrayWeight = {
      asINDArrayWeight(implicitApplyRest(partialApplyData(weightFactory.newInstance, dataParameter(data))))
    }
  }
  trait INDArrayOptimizerApi {

    val weight: INDArrayWeight

    val originalDelta: Nd4jArray
    def delta: Nd4jArray = originalDelta
  }
  type INDArrayOptimizer <: INDArrayOptimizerApi

  @(inject @getter)
  val optimizerFactory: Factory[INDArrayOptimizer]

  @(inject @getter)
  val partialApplyWeight: PartialApply[optimizerFactory.Constructor, Witness.`"weight"`.T]

  @(inject @getter)
  val weightParameter: INDArrayWeight <:< partialApplyWeight.Parameter

  @(inject @getter)
  val partialApplyOriginalDelta: PartialApply[partialApplyWeight.Rest, Witness.`"originalDelta"`.T]

  @(inject @getter)
  val originalDeltaParameter: Nd4jArray <:< partialApplyOriginalDelta.Parameter

  @(inject @getter)
  val weightFactory: Factory[INDArrayWeight]

  @(inject @getter)
  val partialApplyData: PartialApply[weightFactory.Constructor, Witness.`"data"`.T]

  @(inject @getter)
  val dataParameter: Nd4jArray <:< partialApplyData.Parameter

  trait ImplicitsApi extends super.ImplicitsApi {

    @inline
    implicit def liftINDArray[A](
        implicit typeClass: LowPriorityLift.Aux[A, Nd4jArray, Nd4jArray]): Lift.Aux[A, Nd4jArray, Nd4jArray] =
      typeClass

    @inline
    implicit def liftINDArrayWeight[SubtypeOfOptimizer](
                                                      implicit implicitApplyRest: ImplicitApply.Aux[partialApplyOriginalDelta.Rest, SubtypeOfOptimizer],
                                                      constraint: SubtypeOfOptimizer <:< INDArrayOptimizer): Lift.Aux[INDArrayWeight, Nd4jArray, Nd4jArray] = {
      new Lift[INDArrayWeight] {
        override type Data = Nd4jArray
        override type Delta = Nd4jArray

        @inline
        override def apply(weight: INDArrayWeight): Do[Tape[Nd4jArray, Nd4jArray]] = {
          import weight._
          Do.now(Tape(data, backward[SubtypeOfOptimizer]))
        }
      }
    }

    implicit object INDArrayLoss extends Loss[Nd4jArray, Nd4jArray] {
      override def deltaLoss(data: Nd4jArray): Do[Nd4jArray] = Do.now(Nd4j.ones(data.shape(): _*))
    }

    @inline
    implicit def `INDArray+INDArray`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions.+.Case.Aux[Do[Tape[Nd4jArray, Nd4jArray]],
                                                                      Do[Tape[Nd4jArray, Nd4jArray]],
                                                                      Do[Tape[Nd4jArray, Nd4jArray]]] = {
      polyFunctions.+.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0: Nd4jArray, data1: Nd4jArray) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = {
              val newShape = autoBroadcastShape(data0.shape(), data1.shape())
              data0.broadcast(newShape: _*) + data1.broadcast(newShape: _*)
            }
            val computeBackward = { outputDelta: Nd4jArray =>
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
        executionContext: ExecutionContext): polyFunctions.+.Case.Aux[Do[Tape[Nd4jArray, Nd4jArray]],
                                                                      Do[Tape[scala.Double, scala.Double]],
                                                                      Do[Tape[Nd4jArray, Nd4jArray]]] = {
      polyFunctions.+.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0: Nd4jArray, data1: scala.Double) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = data0 + data1
            val computeBackward = { outputDelta: Nd4jArray =>
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
                                                                      Do[Tape[Nd4jArray, Nd4jArray]],
                                                                      Do[Tape[Nd4jArray, Nd4jArray]]] = {
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
        executionContext: ExecutionContext): polyFunctions.-.Case.Aux[Do[Tape[Nd4jArray, Nd4jArray]],
                                                                      Do[Tape[Nd4jArray, Nd4jArray]],
                                                                      Do[Tape[Nd4jArray, Nd4jArray]]] = {
      polyFunctions.-.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0: Nd4jArray, data1: Nd4jArray) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = {
              val newShape = autoBroadcastShape(data0.shape(), data1.shape())
              data0.broadcast(newShape: _*) - data1.broadcast(newShape: _*)
            }
            val computeBackward = { outputDelta: Nd4jArray =>
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
        executionContext: ExecutionContext): polyFunctions.-.Case.Aux[Do[Tape[Nd4jArray, Nd4jArray]],
                                                                      Do[Tape[scala.Double, scala.Double]],
                                                                      Do[Tape[Nd4jArray, Nd4jArray]]] = {
      polyFunctions.-.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0: Nd4jArray, data1: scala.Double) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = data0 - data1
            val computeBackward = { outputDelta: Nd4jArray =>
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
                                                                      Do[Tape[Nd4jArray, Nd4jArray]],
                                                                      Do[Tape[Nd4jArray, Nd4jArray]]] = {
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
        executionContext: ExecutionContext): polyFunctions.*.Case.Aux[Do[Tape[Nd4jArray, Nd4jArray]],
                                                                      Do[Tape[Nd4jArray, Nd4jArray]],
                                                                      Do[Tape[Nd4jArray, Nd4jArray]]] = {
      polyFunctions.*.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0: Nd4jArray, data1: Nd4jArray) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = {
              val newShape = autoBroadcastShape(data0.shape(), data1.shape())
              data0.broadcast(newShape: _*) * data1.broadcast(newShape: _*)
            }
            val computeBackward = { outputDelta: Nd4jArray =>
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
        executionContext: ExecutionContext): polyFunctions.*.Case.Aux[Do[Tape[Nd4jArray, Nd4jArray]],
                                                                      Do[Tape[scala.Double, scala.Double]],
                                                                      Do[Tape[Nd4jArray, Nd4jArray]]] = {
      polyFunctions.*.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0: Nd4jArray, data1: scala.Double) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = data0 * data1
            val computeBackward = { outputDelta: Nd4jArray =>
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
                                                                      Do[Tape[Nd4jArray, Nd4jArray]],
                                                                      Do[Tape[Nd4jArray, Nd4jArray]]] = {
      polyFunctions.*.at { (operand0, operand1) =>
        `INDArray*Double`.apply(operand1, operand0)

      }
    }

    @inline
    implicit def `INDArray/INDArray`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions./.Case.Aux[Do[Tape[Nd4jArray, Nd4jArray]],
                                                                      Do[Tape[Nd4jArray, Nd4jArray]],
                                                                      Do[Tape[Nd4jArray, Nd4jArray]]] = {
      polyFunctions./.at { (operand0, operand1) =>
        operand0 * reciprocal(operand1)
      }
    }

    @inline
    implicit def `INDArray/Double`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions./.Case.Aux[Do[Tape[Nd4jArray, Nd4jArray]],
                                                                      Do[Tape[scala.Double, scala.Double]],
                                                                      Do[Tape[Nd4jArray, Nd4jArray]]] = {
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
                                                                      Do[Tape[Nd4jArray, Nd4jArray]],
                                                                      Do[Tape[Nd4jArray, Nd4jArray]]] = {
      polyFunctions./.at { (operand0, operand1) =>
        operand0 * reciprocal(operand1)
      }
    }

    @inline
    implicit def `max(INDArray,Double)`(
        implicit fullName: sourcecode.FullName,
        caller: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): polyFunctions.max.Case.Aux[Do[Tape[Nd4jArray, Nd4jArray]],
                                                                        Do[Tape[scala.Double, scala.Double]],
                                                                        Do[Tape[Nd4jArray, Nd4jArray]]] = {
      polyFunctions.max.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0: Nd4jArray, data1: scala.Double) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = Transforms.max(data0, data1)
            val computeBackward = { outputDelta: Nd4jArray =>
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
        executionContext: ExecutionContext): polyFunctions.min.Case.Aux[Do[Tape[Nd4jArray, Nd4jArray]],
                                                                        Do[Tape[scala.Double, scala.Double]],
                                                                        Do[Tape[Nd4jArray, Nd4jArray]]] = {
      polyFunctions.min.at { (operand0, operand1) =>
        tapefactories.Binary.doTape(operand0, operand1) { (data0: Nd4jArray, data1: scala.Double) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = Transforms.min(data0, data1)
            val computeBackward = { outputDelta: Nd4jArray =>
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
      : polyFunctions.exp.Case.Aux[Do[Tape[Nd4jArray, Nd4jArray]], Do[Tape[Nd4jArray, Nd4jArray]]] = {
      polyFunctions.exp.at { operand =>
        tapefactories.Unary.doTape(operand) { (data: Nd4jArray) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = Transforms.exp(data)
            val computeBackward = { outputDelta: Nd4jArray =>
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
      : polyFunctions.log.Case.Aux[Do[Tape[Nd4jArray, Nd4jArray]], Do[Tape[Nd4jArray, Nd4jArray]]] = {
      polyFunctions.log.at { operand =>
        tapefactories.Unary.doTape(operand) { (data: Nd4jArray) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = Transforms.log(data)
            val computeBackward = { outputDelta: Nd4jArray =>
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
      : polyFunctions.abs.Case.Aux[Do[Tape[Nd4jArray, Nd4jArray]], Do[Tape[Nd4jArray, Nd4jArray]]] = {
      polyFunctions.abs.at { operand =>
        tapefactories.Unary.doTape(operand) { (data: Nd4jArray) =>
          throwableMonadic[Task] {
            jumpTask().each
            val outputData = Transforms.abs(data)
            val computeBackward = { outputDelta: Nd4jArray =>
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
