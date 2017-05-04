package com.thoughtworks.deeplearning.differentiable

import java.util.logging.{Level, Logger}

import com.thoughtworks.deeplearning.LogRecords.{UncaughtExceptionDuringBackward, WeightIsUpdating}
import com.thoughtworks.deeplearning.PolyFunctions.PolyMethods
import com.thoughtworks.deeplearning.TapeTask.Trainable
import com.thoughtworks.deeplearning.ToTapeTask.LowPriorityToTapeTask
import com.thoughtworks.deeplearning.differentiable.double.DoubleTape
import com.thoughtworks.deeplearning._
import com.thoughtworks.raii.future.Do
import com.thoughtworks.raii.future.Do._
import com.thoughtworks.raii.transformers.ResourceFactoryT
import org.nd4j.linalg.api.ndarray.INDArray
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
import com.thoughtworks.raii.ownership.{Borrowing, garbageCollectable}
import shapeless.the

import scala.concurrent.ExecutionContext
import scala.util.{Failure, Success, Try}
import scalaz.{Monoid, Semigroup, \/, \/-}
import scalaz.concurrent.{Future, Task}
import scalaz.syntax.all._

object indarray {
  private[deeplearning] type INDArrayTape = Borrowing[Tape.Aux[INDArray, INDArray]]

  trait Optimizer {

    protected def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = delta

    final def updateINDArray(oldValue: INDArray, delta: INDArray): INDArray = {
      oldValue - currentDelta(oldValue, delta)
    }
  }

  object Optimizer {

    trait LearningRate extends Optimizer {

      protected def currentLearningRate(): Double

      override def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = delta * currentLearningRate()
    }

    trait L1Regularization extends Optimizer {
      protected def l1Regularization: Double

      override def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = {
        super.currentDelta(oldValue, delta + sign(oldValue) * l1Regularization)
      }
    }

    trait L2Regularization extends Optimizer {
      protected def l2Regularization: Double

      override def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = {
        super.currentDelta(oldValue, delta + oldValue * l2Regularization)
      }
    }

    trait Momentum extends Optimizer {
      protected def mu(): Double = 0.9

      private var v: Option[INDArray] = None

      override def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = {
        val vValue: INDArray = v.getOrElse(Nd4j.zeros(delta.shape: _*))
        v = Some(
          super.currentDelta(oldValue, delta) + vValue * mu()
        )
        v.get
      }
    }

    trait NesterovMomentum extends Optimizer {
      protected def mu(): Double = 0.9

      private var v: Option[INDArray] = None

      override def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = {
        val vValue: INDArray = v.getOrElse(Nd4j.zeros(delta.shape: _*))
        val vPre = vValue
        v = Some(
          super.currentDelta(oldValue, delta) + vValue * mu()
        )

        vPre * (-mu()) + v.get * (1 + mu())
      }
    }

    trait Adagrad extends Optimizer {

      protected def eps(): Double = 1e-4

      private var cache: Option[INDArray] = None

      override def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = {
        val cacheValue = cache.getOrElse(Nd4j.zeros(delta.shape: _*))
        cache = Some(cacheValue + delta * delta)
        super.currentDelta(oldValue, delta) / (sqrt(cache.get) + eps)
      }
    }

    trait RMSprop extends Optimizer {

      protected def decayRate(): Double = 0.99

      protected def eps(): Double = 1e-4

      private var cache: Option[INDArray] = None

      override def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = {
        val cacheValue = cache.getOrElse(Nd4j.zeros(delta.shape: _*))
        cache = Some(cacheValue * decayRate + delta * delta * (1 - decayRate))
        super.currentDelta(oldValue, delta) / (sqrt(cache.get) + eps)
      }
    }

    trait Adam extends Optimizer {

      protected def beta1 = 0.9

      protected def beta2 = 0.999

      protected def eps(): Double = 1e-8

      private var m: Option[INDArray] = None

      private var v: Option[INDArray] = None

      private var times: Int = 0

      override def currentDelta(oldValue: INDArray, delta: INDArray): INDArray = {

        val mValue = m.getOrElse(Nd4j.zeros(delta.shape: _*))

        m = Some(
          mValue * beta1 + delta * (1 - beta1)
        )

        val vValue = v.getOrElse(Nd4j.zeros(delta.shape: _*))

        v = Some(
          vValue * beta2 + delta * delta * (1 - beta2)
        )

        times += 1

        val coef1 = 1 - math.pow(beta1, times)

        val coef2 = math.sqrt(1 - math.pow(beta2, times))

        super.currentDelta(oldValue, m.get * (coef2 / coef1)) / (sqrt(v.get) + eps)
      }
    }

  }

  final case class Weight(var data: INDArray)(implicit optimizerFactory: OptimizerFactory,
                                              logger: Logger = Logger.getGlobal,
                                              fullName: sourcecode.FullName,
                                              className: Caller[_],
                                              methodName: sourcecode.Name)
      extends Tape {
    private val optimizer: Optimizer = optimizerFactory.indarrayOptimizer(this)
    override type Data = INDArray
    override type Delta = INDArray

    override def backward(outputDelta: Do[_ <: Delta]): Future[Unit] = {
      import com.thoughtworks.raii.transformers.ResourceFactoryT.resourceFactoryTMonad
      val Do(resourceFactoryTFuture) = outputDelta
      val resourceFactoryT: ResourceFactoryT[Future, Try[Delta]] = ResourceFactoryT(resourceFactoryTFuture)

      val tryTRAIIFuture: ResourceFactoryT[Future, Try[Unit]] = resourceFactoryT.map { tryDelta: Try[Delta] =>
        tryDelta.map { delta =>
          synchronized {
            if (logger.isLoggable(Level.FINER)) {
              logger.log(WeightIsUpdating(data, delta))
            }
            data = optimizer.updateINDArray(data, delta)
          }
        }
      }
      ResourceFactoryT.run(tryTRAIIFuture).flatMap {
        case Failure(e) =>
          logger.log(UncaughtExceptionDuringBackward(e))
          Future.now(())
        case Success(()) =>
          Future.now(())
      }
    }
  }

  object OptimizerFactory {
    implicit def shared(implicit optimizer: Optimizer): OptimizerFactory = new OptimizerFactory {
      override def indarrayOptimizer(weight: Weight): Optimizer = optimizer
    }
  }

  trait OptimizerFactory {
    def indarrayOptimizer(weight: Weight): Optimizer
  }

  // TODO: Add a test for this method and auto-broadcasting on n-dimension arrays for n > 2
  private[differentiable] def sumAs(outputDeltaValue: INDArray, shape: Array[Int]) = {
    val singleElementDimension = (shape: Seq[Int]).view.zip(outputDeltaValue.shape).zipWithIndex.collect {
      case ((1, originSize), dimension) if originSize > 1 => dimension
    }
    if (singleElementDimension.isEmpty) {
      outputDeltaValue
    } else {
      outputDeltaValue.sum(singleElementDimension.force: _*).reshape(shape: _*)
    }
  }

  private[differentiable] def autoBroadcastShape(shape1: Array[Int], shape2: Array[Int]) = {
    require(shape1.length == shape2.length)
    shape1.zip(shape2).map {
      case (1, bSize) => bSize
      case (aSize, 1) => aSize
      case (aSize, bSize) if aSize == bSize => aSize
    }
  }

  object implicits {
    import com.thoughtworks.deeplearning.differentiable.double
    import com.thoughtworks.deeplearning.differentiable.double.implicits._
    import com.thoughtworks.deeplearning.differentiable.double.DoubleTape
    import com.thoughtworks.deeplearning.TapeTaskFactory.BinaryTapeTaskFactory.semigroupBinaryTapeTaskFactory
    import com.thoughtworks.deeplearning.TapeTaskFactory.UnaryTapeTaskFactory.semigroupUnaryTapeTaskFactory

    private implicit object INDArraySemigroup extends Semigroup[INDArray] {
      override def append(f1: INDArray, f2: => INDArray): INDArray = f1 + f2
    }

    @inline
    implicit def liftINDArray[A](
        implicit typeClass: LowPriorityToTapeTask.Aux[A, INDArray, INDArray]): ToTapeTask.Aux[A, INDArray, INDArray] =
      typeClass

    implicit final class INDArrayToWeightOps(value: INDArray) {
      def toWeight(implicit optimizerFactory: OptimizerFactory,
                   logger: Logger = Logger.getGlobal,
                   fullName: sourcecode.FullName,
                   className: Caller[_],
                   methodName: sourcecode.Name): Do[INDArrayTape] = {
        val myWeight = garbageCollectable(Weight(value))
        Do.now(myWeight)
      }
    }

    implicit object INDArrayTrainable extends Trainable[INDArray, INDArray] {
      override def apply(data: INDArray): Do[_ <: INDArray] = Do.now(Nd4j.ones(data.shape(): _*))
    }

    private def jump()(implicit executionContext: ExecutionContext): Task[Unit] = {
      Task.async { handler: ((Throwable \/ Unit) => Unit) =>
        executionContext.execute {
          new Runnable {
            override def run(): Unit = handler(\/-(()))
          }
        }
      }
    }

    //TODO:耗时操作需要 Task.jump().await
    @inline
    implicit def `INDArray+INDArray`(implicit logger: Logger = Logger.getGlobal,
                                     fullName: sourcecode.FullName,
                                     className: Caller[_],
                                     methodName: sourcecode.Name,
                                     executionContext: ExecutionContext)
      : PolyMethods.+.Case.Aux[Do.Covariant[INDArrayTape], Do.Covariant[INDArrayTape], Do[INDArrayTape]] = {
      PolyMethods.+.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0: INDArray, data1: INDArray) =>
          throwableMonadic[Task] {
            //Task.jump().await
            //jump().await
            jump().each
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
    implicit def `INDArray+Double`(implicit logger: Logger = Logger.getGlobal,
                                   fullName: sourcecode.FullName,
                                   className: Caller[_],
                                   methodName: sourcecode.Name,
                                   executionContext: ExecutionContext)
      : PolyMethods.+.Case.Aux[Do.Covariant[INDArrayTape], Do.Covariant[DoubleTape], Do[INDArrayTape]] = {
      PolyMethods.+.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0: INDArray, data1: Double) =>
          throwableMonadic[Task] {
            jump().each
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
    implicit def `Double+INDArray`(implicit logger: Logger = Logger.getGlobal,
                                   fullName: sourcecode.FullName,
                                   className: Caller[_],
                                   methodName: sourcecode.Name,
                                   executionContext: ExecutionContext)
      : PolyMethods.+.Case.Aux[Do.Covariant[DoubleTape], Do.Covariant[INDArrayTape], Do[INDArrayTape]] = {
      PolyMethods.+.at { (operand0, operand1) =>
        `INDArray+Double`.apply(operand1, operand0)
      }
    }

    @inline
    implicit def `INDArray-INDArray`(implicit logger: Logger = Logger.getGlobal,
                                     fullName: sourcecode.FullName,
                                     className: Caller[_],
                                     methodName: sourcecode.Name,
                                     executionContext: ExecutionContext)
      : PolyMethods.-.Case.Aux[Do.Covariant[INDArrayTape], Do.Covariant[INDArrayTape], Do[INDArrayTape]] = {
      PolyMethods.-.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0: INDArray, data1: INDArray) =>
          throwableMonadic[Task] {
            jump().each
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
    implicit def `INDArray-Double`(implicit logger: Logger = Logger.getGlobal,
                                   fullName: sourcecode.FullName,
                                   className: Caller[_],
                                   methodName: sourcecode.Name,
                                   executionContext: ExecutionContext)
      : PolyMethods.-.Case.Aux[Do.Covariant[INDArrayTape], Do.Covariant[DoubleTape], Do[INDArrayTape]] = {
      PolyMethods.-.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0: INDArray, data1: Double) =>
          throwableMonadic[Task] {
            jump().each
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
    implicit def `Double-INDArray`(implicit logger: Logger = Logger.getGlobal,
                                   fullName: sourcecode.FullName,
                                   className: Caller[_],
                                   methodName: sourcecode.Name,
                                   executionContext: ExecutionContext)
      : PolyMethods.-.Case.Aux[Do.Covariant[DoubleTape], Do.Covariant[INDArrayTape], Do[INDArrayTape]] = {
      PolyMethods.-.at { (operand0, operand1) =>
        `INDArray+Double`.apply(negative(operand1), operand0)
//        TapeTaskFactory.binary(operand0, operand1) { (data0: Double, data1: INDArray) =>
//          throwableMonadic[Task] {
//            jump().each
//            val outputData = -data1 + data0
//            val computeBackward = { outputDelta: INDArray =>
//              val delta0Future = throwableMonadic[Do] {
//                Do.jump().each
//                outputDelta.sumT
//              }
//              val delta1Future = throwableMonadic[Do] {
//                Do.jump().each
//                -outputDelta
//              }
//              (delta0Future, delta1Future)
//            }
//            (outputData, computeBackward)
//          }
//        }
      }
    }

    @inline
    implicit def `INDArray*INDArray`(implicit logger: Logger = Logger.getGlobal,
                                     fullName: sourcecode.FullName,
                                     className: Caller[_],
                                     methodName: sourcecode.Name,
                                     executionContext: ExecutionContext)
      : PolyMethods.*.Case.Aux[Do.Covariant[INDArrayTape], Do.Covariant[INDArrayTape], Do[INDArrayTape]] = {
      PolyMethods.*.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0: INDArray, data1: INDArray) =>
          throwableMonadic[Task] {
            jump().each
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
    implicit def `INDArray*Double`(implicit logger: Logger = Logger.getGlobal,
                                   fullName: sourcecode.FullName,
                                   className: Caller[_],
                                   methodName: sourcecode.Name,
                                   executionContext: ExecutionContext)
      : PolyMethods.*.Case.Aux[Do.Covariant[INDArrayTape], Do.Covariant[DoubleTape], Do[INDArrayTape]] = {
      PolyMethods.*.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0: INDArray, data1: Double) =>
          throwableMonadic[Task] {
            jump().each
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
    implicit def `Double*INDArray`(implicit logger: Logger = Logger.getGlobal,
                                   fullName: sourcecode.FullName,
                                   className: Caller[_],
                                   methodName: sourcecode.Name,
                                   executionContext: ExecutionContext)
      : PolyMethods.*.Case.Aux[Do.Covariant[DoubleTape], Do.Covariant[INDArrayTape], Do[INDArrayTape]] = {
      PolyMethods.*.at { (operand0, operand1) =>
        `INDArray*Double`.apply(operand1, operand0)
      }
    }

    @inline
    implicit def `INDArray/INDArray`(implicit logger: Logger = Logger.getGlobal,
                                     fullName: sourcecode.FullName,
                                     className: Caller[_],
                                     methodName: sourcecode.Name,
                                     executionContext: ExecutionContext)
      : PolyMethods./.Case.Aux[Do.Covariant[INDArrayTape], Do.Covariant[INDArrayTape], Do[INDArrayTape]] = {
      PolyMethods./.at { (operand0, operand1) =>
        `INDArray*INDArray`.apply(operand0, reciprocal(operand1))
      }
    }

    @inline
    implicit def `INDArray/Double`(implicit logger: Logger = Logger.getGlobal,
                                   fullName: sourcecode.FullName,
                                   className: Caller[_],
                                   methodName: sourcecode.Name,
                                   executionContext: ExecutionContext)
      : PolyMethods./.Case.Aux[Do.Covariant[INDArrayTape], Do.Covariant[DoubleTape], Do[INDArrayTape]] = {
      PolyMethods./.at { (operand0, operand1) =>
        `INDArray*Double`.apply(operand0, double.implicits.reciprocal(operand1))
      }
    }

    @inline
    implicit def `Double/INDArray`(implicit logger: Logger = Logger.getGlobal,
                                   fullName: sourcecode.FullName,
                                   className: Caller[_],
                                   methodName: sourcecode.Name,
                                   executionContext: ExecutionContext)
      : PolyMethods./.Case.Aux[Do.Covariant[DoubleTape], Do.Covariant[INDArrayTape], Do[INDArrayTape]] = {
      PolyMethods./.at { (operand0, operand1) =>
        `Double*INDArray`.apply(operand0, reciprocal(operand1))
      }
    }

    @inline
    implicit def `max(INDArray,Double)`(implicit logger: Logger = Logger.getGlobal,
                                        fullName: sourcecode.FullName,
                                        className: Caller[_],
                                        methodName: sourcecode.Name,
                                        executionContext: ExecutionContext)
      : PolyFunctions.max.Case.Aux[Do.Covariant[INDArrayTape], Do.Covariant[DoubleTape], Do[INDArrayTape]] = {
      PolyFunctions.max.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0: INDArray, data1: Double) =>
          throwableMonadic[Task] {
            jump().each
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
    implicit def `min(INDArray,Double)`(implicit logger: Logger = Logger.getGlobal,
                                        fullName: sourcecode.FullName,
                                        className: Caller[_],
                                        methodName: sourcecode.Name,
                                        executionContext: ExecutionContext)
      : PolyFunctions.min.Case.Aux[Do.Covariant[INDArrayTape], Do.Covariant[DoubleTape], Do[INDArrayTape]] = {
      PolyFunctions.min.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0: INDArray, data1: Double) =>
          throwableMonadic[Task] {
            jump().each
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
    implicit def `exp(INDArray)`(implicit logger: Logger = Logger.getGlobal,
                                 fullName: sourcecode.FullName,
                                 className: Caller[_],
                                 methodName: sourcecode.Name,
                                 executionContext: ExecutionContext)
      : PolyFunctions.exp.Case.Aux[Do.Covariant[INDArrayTape], Do[INDArrayTape]] = {
      PolyFunctions.exp.at { operand =>
        TapeTaskFactory.unary(operand) { (data: INDArray) =>
          throwableMonadic[Task] {
            jump().each
            val outputData = Transforms.exp(data)
            val computeBackward = { outputDelta: INDArray =>
              throwableMonadic[Do] {
                Do.jump().each
                data * outputDelta
              }
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `log(INDArray)`(implicit logger: Logger = Logger.getGlobal,
                                 fullName: sourcecode.FullName,
                                 className: Caller[_],
                                 methodName: sourcecode.Name,
                                 executionContext: ExecutionContext)
      : PolyFunctions.log.Case.Aux[Do.Covariant[INDArrayTape], Do[INDArrayTape]] = {
      PolyFunctions.log.at { operand =>
        TapeTaskFactory.unary(operand) { (data: INDArray) =>
          throwableMonadic[Task] {
            jump().each
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
    implicit def `abs(INDArray)`(implicit logger: Logger = Logger.getGlobal,
                                 fullName: sourcecode.FullName,
                                 className: Caller[_],
                                 methodName: sourcecode.Name,
                                 executionContext: ExecutionContext)
      : PolyFunctions.abs.Case.Aux[Do.Covariant[INDArrayTape], Do[INDArrayTape]] = {
      PolyFunctions.abs.at { operand =>
        TapeTaskFactory.unary(operand) { (data: INDArray) =>
          throwableMonadic[Task] {
            jump().each
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

    @inline
    def negative(operand: Do.Covariant[INDArrayTape])(
        implicit logger: Logger = Logger.getGlobal,
        fullName: sourcecode.FullName,
        className: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): Do.Covariant[INDArrayTape] = {
      TapeTaskFactory.unary(operand) { data =>
        throwableMonadic[Task] {
          jump().each
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

    @inline
    def reciprocal(operand: Do.Covariant[INDArrayTape])(
        implicit logger: Logger = Logger.getGlobal,
        fullName: sourcecode.FullName,
        className: Caller[_],
        methodName: sourcecode.Name,
        executionContext: ExecutionContext): Do.Covariant[INDArrayTape] = {
      TapeTaskFactory.unary(operand) { data: INDArray =>
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

    implicit final class DifferentiableINDArrayOps[From](from: From)(
        implicit lift: ToTapeTask.Aux[From, INDArray, INDArray],
        logger: Logger = Logger.getGlobal,
        fullName: sourcecode.FullName,
        methodName: sourcecode.Name,
        className: Caller[_],
        executionContext: ExecutionContext) {
      private val operand: Do.Covariant[INDArrayTape] = lift(from)
      @inline
      def unary_- : Do[INDArrayTape] = {
        TapeTaskFactory.unary(operand) { data =>
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
  }
}
