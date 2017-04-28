package com.thoughtworks.deeplearning.differentiable

import java.util.logging.{Level, Logger}

import com.thoughtworks.deeplearning.LogRecords.{FloatWeightTracker, UncaughtExceptionDuringBackward}
import com.thoughtworks.deeplearning.{Tape, TapeTaskFactory, ToTapeTask}
import com.thoughtworks.deeplearning.TapeTaskFactory.BinaryTapeTaskFactory
import com.thoughtworks.deeplearning.PolyFunctions._
import com.thoughtworks.deeplearning.TapeTask.Trainable
import com.thoughtworks.deeplearning.differentiable.float.{Optimizer, OptimizerFactory, Weight}
import com.thoughtworks.raii.future.Do
import com.thoughtworks.raii.transformers.ResourceFactoryT
import com.thoughtworks.deeplearning.TapeTaskFactory.BinaryTapeTaskFactory.monoidBinaryTapeTaskFactory
import com.thoughtworks.deeplearning.differentiable.float.implicits.FloatMonoid
import shapeless.the

import scala.util.{Failure, Success, Try}
import scalaz.Monoid
import scalaz.concurrent.{Future, Task}
import scalaz.syntax.all._

/**
  * @author 张志豪 (izhangzhihao) &lt;izhangzhihao@hotmail.com&gt;
  */
object int {
  private[deeplearning] type IntTape = Tape.Aux[Int, Float]

  val Optimizer = float.Optimizer

  final case class Weight(var data: Int)(implicit optimizerFactory: OptimizerFactory,
                                         logger: Logger = Logger.getGlobal)
      extends Tape {
    private val optimizer: Optimizer = optimizerFactory.intOptimizer(this)

    override type Data = Int
    override type Delta = Float

    override def backward(deltaFuture: Do[_ <: Delta]): Future[Unit] = {
      import com.thoughtworks.raii.transformers.ResourceFactoryT.resourceFactoryTMonad

      val Do(resourceFactoryTFuture) = deltaFuture

      val resourceFactoryT: ResourceFactoryT[Future, Try[Delta]] = ResourceFactoryT(resourceFactoryTFuture)

      val tryTRAIIFuture: ResourceFactoryT[Future, Try[Unit]] = resourceFactoryT.map { tryDelta: Try[Delta] =>
        tryDelta.map { delta =>
          synchronized {
            if (logger.isLoggable(Level.FINER)) {
              logger.log(FloatWeightTracker(s"weight: $data, delta:$delta"))
            }
            data = optimizer.updateFloat(data.toFloat, delta).toInt
          }
        }
      }

      ResourceFactoryT.run(tryTRAIIFuture).flatMap {
        case Failure(e) =>
          logger.log(UncaughtExceptionDuringBackward(e, "An exception raised during backward"))
          Future.now(())
        case Success(()) =>
          Future.now(())
      }
    }
  }

  object OptimizerFactory {
    implicit def shared(implicit optimizer: Optimizer): OptimizerFactory = new OptimizerFactory {
      override def intOptimizer(weight: Weight): Optimizer = optimizer
    }
  }

  trait OptimizerFactory {
    def intOptimizer(weight: Weight): Optimizer
  }

  object implicits {
    private implicit object IntMonoid extends Monoid[Int] {
      override def zero: Int = 0

      override def append(f1: Int, f2: => Int): Int = f1 + f2
    }

    @inline
    implicit def liftInt[A](implicit typeClass: ToTapeTask.Aux[A, Int, Float]): ToTapeTask.Aux[A, Int, Float] =
      typeClass

    implicit final class IntToWeightOps(value: Int) {
      def toWeight(implicit optimizerFactory: OptimizerFactory, logger: Logger = Logger.getGlobal): Weight = {
        Weight(value)
      }
    }

    implicit object IntTrainable extends Trainable[Int, Float] {
      override def apply(data: Int): Do[Float] = Do.now(1)
    }

    @inline
    implicit def `Int+Int`(implicit logger: Logger = Logger.getGlobal)
      : PolyMethods.+.Case.Aux[Do.Covariant[IntTape], Do.Covariant[IntTape], Do[IntTape]] = {
      PolyMethods.+.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val outputData = data0 + data1
            val computeBackward = { outputDelta: Float =>
              val delta0Future = Do.now(outputDelta)
              val delta1Future = Do.now(outputDelta)
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `Int-Int`(implicit logger: Logger = Logger.getGlobal)
      : PolyMethods.-.Case.Aux[Do.Covariant[IntTape], Do.Covariant[IntTape], Do[IntTape]] = {
      PolyMethods.-.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val outputData = data0 - data1
            val computeBackward = { outputDelta: Float =>
              val delta0Future = Do.now(outputDelta)
              val delta1Future = Do.delay(-outputDelta)
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `Int*Int`(implicit logger: Logger = Logger.getGlobal)
      : PolyMethods.*.Case.Aux[Do.Covariant[IntTape], Do.Covariant[IntTape], Do[IntTape]] = {
      PolyMethods.*.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val outputData = data0 * data1
            val computeBackward = { outputDelta: Float =>
              val delta0Future = Do.delay(outputDelta * data1)
              val delta1Future = Do.delay(outputDelta * data0)
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    @inline
    implicit def `Int/Int`(implicit logger: Logger = Logger.getGlobal)
      : PolyMethods./.Case.Aux[Do.Covariant[IntTape], Do.Covariant[IntTape], Do[IntTape]] = {
      PolyMethods./.at { (operand0, operand1) =>
        TapeTaskFactory.binary(operand0, operand1) { (data0, data1) =>
          Task.delay {
            val outputData = data0 / data1
            val computeBackward = { outputDelta: Float =>
              val delta0Future = Do.delay(outputDelta / data1)
              val delta1Future = Do.delay(-data0 * outputDelta / (data1 * data1))
              (delta0Future, delta1Future)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

    implicit final class DifferentiableIntOps[From](from: From)(implicit lift: ToTapeTask.Aux[From, Int, Float],
                                                                logger: Logger = Logger.getGlobal) {
      private val operand: Do.Covariant[IntTape] = lift(from)
      @inline
      def unary_- : Do[IntTape] = {
        TapeTaskFactory.unary(operand) { data =>
          Task.delay {
            val outputData = -data
            val computeBackward = { outputDelta: Float =>
              Do.delay(-outputDelta)
            }
            (outputData, computeBackward)
          }
        }
      }
    }

  }

}
