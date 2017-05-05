package com.thoughtworks.deeplearning

import java.util.logging.{Level, Logger}

import com.thoughtworks.deeplearning.LogRecords.{DeltaAccumulatorIsUpdating, UncaughtExceptionDuringBackward}
import com.thoughtworks.deeplearning.Tape.Aux
import com.thoughtworks.raii._

import scalaz.{-\/, @@, Applicative, Monoid, Semigroup, \/, \/-}
import scalaz.concurrent.{Future, Task}
import com.thoughtworks.raii.Shared._
import com.thoughtworks.raii.future.Do
import com.thoughtworks.raii.future.Do.AsyncReleasable
import com.thoughtworks.raii.resourcet.{ResourceT, Releasable}
import com.thoughtworks.tryt.TryT

import scala.language.existentials
import scala.util.{Failure, Success, Try}
import scala.util.control.NoStackTrace
import scalaz.Tags.Parallel
import scalaz.syntax.all._
import scala.util.control.NonFatal
import com.thoughtworks.raii.ownership._
import com.thoughtworks.raii.ownership.implicits._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object TapeTaskFactory {

  @inline
  def binary[Data0, Delta0, Data1, Delta1, OutputData, OutputDelta](
      operand0: Do[_ <: Borrowing[Tape.Aux[Data0, Delta0]]],
      operand1: Do[_ <: Borrowing[Tape.Aux[Data1, Delta1]]])(
      computeForward: (Data0, Data1) => Task[(OutputData, OutputDelta => (Do[_ <: Delta0], Do[_ <: Delta1]))])(
      implicit binaryTapeTaskFactory: BinaryTapeTaskFactory[OutputData, OutputDelta],
      logger: Logger,
      fullName: sourcecode.FullName,
      methodName: sourcecode.Name,
      className: Caller[_]): Do[Borrowing[Tape.Aux[OutputData, OutputDelta]]] = {
    binaryTapeTaskFactory(operand0, operand1)(computeForward)
  }

  @inline
  def unary[Data, Delta, OutputData, OutputDelta](operand: Do[_ <: Borrowing[Tape.Aux[Data, Delta]]])(
      computeForward: (Data) => Task[(OutputData, OutputDelta => Do[_ <: Delta])])(
      implicit unaryTapeTaskFactory: UnaryTapeTaskFactory[OutputData, OutputDelta],
      logger: Logger,
      fullName: sourcecode.FullName,
      methodName: sourcecode.Name,
      className: Caller[_]): Do[Borrowing[Tape.Aux[OutputData, OutputDelta]]] = {
    unaryTapeTaskFactory(operand)(computeForward)
  }

  private abstract class Output[OutputData, OutputDelta: Monoid](override val data: OutputData)(
      implicit logger: Logger,
      fullName: sourcecode.FullName,
      methodName: sourcecode.Name,
      className: Caller[_])
      extends Tape
      with Releasable[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]] {

    override def release: Future[Unit] = releaseValue

    override def releaseDependencies: Future[Unit] = Future.now(())

    override final type Delta = OutputDelta
    override final type Data = OutputData

    // TODO: Output should be scoped by `Do.scoped`, not own itself.
    final override def value: Try[this.type Owned this.type] = Success(this.own(this))

    @volatile
    protected var deltaAccumulator: OutputDelta = mzero[OutputDelta]

    final override def backward(deltaFuture: Do[_ <: OutputDelta]): Future[Unit] = {

      import com.thoughtworks.raii.resourcet.ResourceT.resourceFactoryTMonad

      val Do(resourceFactoryTFuture) = deltaFuture

      val resourceFactoryT: ResourceT[Future, Try[OutputDelta]] = ResourceT(resourceFactoryTFuture)

      val tryTRAIIFuture: ResourceT[Future, Try[Unit]] = resourceFactoryT.map { tryDelta: Try[OutputDelta] =>
        tryDelta.map { delta =>
          synchronized {
            if (logger.isLoggable(Level.FINER)) {
              logger.log(DeltaAccumulatorIsUpdating(deltaAccumulator, delta))
            }
            deltaAccumulator |+|= delta
          }
        }
      }

      ResourceT.run(tryTRAIIFuture).flatMap {
        case Failure(e) =>
          logger.log(UncaughtExceptionDuringBackward(e))
          Future.now(())
        case Success(()) =>
          Future.now(())
      }
    }

  }

  trait BinaryTapeTaskFactory[OutputData, OutputDelta] {

    def apply[Data0, Delta0, Data1, Delta1](operand0: Do[_ <: Borrowing[Tape.Aux[Data0, Delta0]]],
                                            operand1: Do[_ <: Borrowing[Tape.Aux[Data1, Delta1]]])(
        computeForward: (Data0, Data1) => Task[(OutputData, OutputDelta => (Do[_ <: Delta0], Do[_ <: Delta1]))])
      : Do[Borrowing[Tape.Aux[OutputData, OutputDelta]]]

  }

  trait UnaryTapeTaskFactory[OutputData, OutputDelta] {

    def apply[Data, Delta](operand: Do[_ <: Borrowing[Tape.Aux[Data, Delta]]])(
        computeForward: (Data) => Task[(OutputData, OutputDelta => Do[_ <: Delta])])
      : Do[Borrowing[Tape.Aux[OutputData, OutputDelta]]]
  }

  object BinaryTapeTaskFactory {

    /** An exception that contains multiple Throwables. */
    final case class MultipleException(throwableSet: Set[Throwable])
        extends Exception("Multiple exceptions found")
        with NoStackTrace {
      override def toString: String = throwableSet.toString()
    }

    implicit def throwableSemigroup = new Semigroup[Throwable] {
      override def append(f1: Throwable, f2: => Throwable): Throwable =
        f1 match {
          case MultipleException(exceptionSet1) =>
            f2 match {
              case MultipleException(exceptionSet2) => MultipleException(exceptionSet1 ++ exceptionSet2)
              case _: Throwable => MultipleException(exceptionSet1 + f2)
            }
          case _: Throwable =>
            f2 match {
              case MultipleException(exceptionSet2) => MultipleException(exceptionSet2 + f1)
              case _: Throwable => MultipleException(Set(f1, f2))
            }
        }
    }

    final class MonoidBinaryTapeTaskFactory[OutputData, OutputDelta: Monoid](implicit logger: Logger,
                                                                             fullName: sourcecode.FullName,
                                                                             methodName: sourcecode.Name,
                                                                             className: Caller[_])
        extends BinaryTapeTaskFactory[OutputData, OutputDelta] {
      @inline
      override def apply[Data0, Delta0, Data1, Delta1](operand0: Do[_ <: Borrowing[Tape.Aux[Data0, Delta0]]],
                                                       operand1: Do[_ <: Borrowing[Tape.Aux[Data1, Delta1]]])(
          computeForward: (Data0, Data1) => Task[(OutputData, OutputDelta => (Do[_ <: Delta0], Do[_ <: Delta1]))])
        : Do[Borrowing[Tape.Aux[OutputData, OutputDelta]]] = {

        import com.thoughtworks.raii.resourcet.ResourceT.resourceFactoryTParallelApplicative
        import com.thoughtworks.raii.future.Do.doParallelApplicative

        val parallelTuple =
          Applicative[Lambda[x => Do[x] @@ Parallel]]
            .tuple2(Parallel(operand0), Parallel(operand1))

        val tuple = Parallel.unwrap(parallelTuple)

        import com.thoughtworks.raii.future.Do.doMonadErrorInstances

        tuple.flatMap { pair =>
          val (upstream0, upstream1) = pair
          val resource: ResourceT[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]] = {
            val futureResourceT: Future[Releasable[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]]] =
              computeForward(upstream0.data, upstream1.data).get.map {
                case left @ -\/(e) =>
                  Releasable.now(Failure(e))
                case right @ \/-((forwardData, computeBackward)) =>
                  new Output[OutputData, OutputDelta](forwardData) {
                    override def releaseValue: Future[Unit] = {
                      val (upstream0DeltaFuture, upstream1DeltaFuture) = computeBackward(deltaAccumulator)
                      Parallel.unwrap {
                        Future.futureParallelApplicativeInstance.apply2(
                          Parallel(upstream0.backward(upstream0DeltaFuture)),
                          Parallel(upstream1.backward(upstream1DeltaFuture))
                        ) { (_: Unit, _: Unit) =>
                          ()
                        }
                      }
                    }
                  }
              }
            ResourceT(futureResourceT)
          }

          val sharedResourceFactoryT: ResourceT[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]] =
            resource.shared

          val ResourceT(future) = sharedResourceFactoryT

          Do(future)
        }
      }
    }

    @inline
    implicit def monoidBinaryTapeTaskFactory[OutputData, OutputDelta: Monoid](
        implicit logger: Logger,
        fullName: sourcecode.FullName,
        methodName: sourcecode.Name,
        className: Caller[_]): BinaryTapeTaskFactory[OutputData, OutputDelta] = {
      new MonoidBinaryTapeTaskFactory[OutputData, OutputDelta]
    }
  }

  object UnaryTapeTaskFactory {
    final class MonoidUnaryTapeTaskFactory[OutputData, OutputDelta: Monoid](implicit logger: Logger,
                                                                            fullName: sourcecode.FullName,
                                                                            methodName: sourcecode.Name,
                                                                            className: Caller[_])
        extends UnaryTapeTaskFactory[OutputData, OutputDelta] {
      @inline
      override def apply[Data, Delta](operand: Do[_ <: Borrowing[Tape.Aux[Data, Delta]]])(
          computeForward: Data => Task[(OutputData, OutputDelta => Do[_ <: Delta])])
        : Do[Borrowing[Tape.Aux[OutputData, OutputDelta]]] = {
        import com.thoughtworks.raii.resourcet.ResourceT._
        import com.thoughtworks.raii.future.Do.doMonadErrorInstances
        operand.flatMap {
          case (upstream) =>
            val resource: ResourceT[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]] = {
              val futureResourceT: Future[Releasable[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]]] =
                computeForward(upstream.data).get.map {
                  case left @ -\/(e) =>
                    Releasable.now(Failure(e))
                  case right @ \/-((forwardData, computeBackward)) =>
                    new Output[OutputData, OutputDelta](forwardData) {
                      override def releaseValue: Future[Unit] = {
                        upstream.backward(computeBackward(deltaAccumulator))
                      }
                    }
                }
              ResourceT(futureResourceT)
            }
            val sharedResourceFactoryT: ResourceT[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]] =
              resource.shared

            val ResourceT(future) = sharedResourceFactoryT

            Do(future)
        }
      }
    }

    @inline
    implicit def monoidUnaryTapeTaskFactory[OutputData, OutputDelta: Monoid](
        implicit logger: Logger,
        fullName: sourcecode.FullName,
        methodName: sourcecode.Name,
        className: Caller[_]): UnaryTapeTaskFactory[OutputData, OutputDelta] = {
      new MonoidUnaryTapeTaskFactory[OutputData, OutputDelta]
    }
  }

}
