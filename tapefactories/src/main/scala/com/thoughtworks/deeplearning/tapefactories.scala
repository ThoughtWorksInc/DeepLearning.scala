package com.thoughtworks.deeplearning

import java.util.logging.{Level, Logger}

import com.thoughtworks.deeplearning.logs.{DeltaAccumulatorIsUpdating, UncaughtExceptionDuringBackward}
import com.thoughtworks.deeplearning.Tape.Aux
import com.thoughtworks.raii._

import scalaz.{-\/, @@, Applicative, Monoid, Semigroup, \/, \/-}
import scalaz.concurrent.{Future, Task}
import com.thoughtworks.raii.shared._
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.covariant.{Releasable, ResourceT}
import com.thoughtworks.tryt.covariant.TryT

import scala.language.existentials
import scala.util.{Failure, Success, Try}
import scala.util.control.NoStackTrace
import scalaz.Tags.Parallel
import scalaz.syntax.all._
import scalaz.std.option._
import scala.util.control.NonFatal
import com.thoughtworks.raii.ownership._
import com.thoughtworks.raii.ownership._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object tapefactories {

  private abstract class MonoidOutput[OutputData, OutputDelta: Monoid](override val data: OutputData)(
      implicit logger: Logger,
      fullName: sourcecode.FullName,
      methodName: sourcecode.Name,
      className: Caller[_])
      extends Tape
      with Releasable[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]] {

    override final type Delta = OutputDelta
    override final type Data = OutputData

    // TODO: MonoidOutput should be scoped by `Do.scoped`, not own itself.
    final override def value: Try[this.type Owned this.type] = Success(this.own(this))

    @volatile
    protected var deltaAccumulator: OutputDelta = mzero[OutputDelta]

    final override def backward(deltaFuture: Do[ OutputDelta]): Future[Unit] = {

      import com.thoughtworks.raii.covariant.ResourceT.resourceTMonad

      val Do(resourceTFuture) = deltaFuture

      val resourceT: ResourceT[Future, Try[OutputDelta]] = ResourceT(resourceTFuture)

      val tryTRAIIFuture: ResourceT[Future, Try[Unit]] = resourceT.map { tryDelta: Try[OutputDelta] =>
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

  private abstract class SemigroupOutput[OutputData, OutputDelta: Semigroup](override val data: OutputData)(
      implicit logger: Logger,
      fullName: sourcecode.FullName,
      methodName: sourcecode.Name,
      className: Caller[_])
      extends Tape
      with Releasable[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]] {

    override final type Delta = OutputDelta
    override final type Data = OutputData
    // TODO: SemigroupOutput should be managed by `Do.managed`, not own itself.
    override def value: Try[this.type Owned this.type] = Success(this.own(this))

    @volatile
    protected var deltaAccumulator: Option[OutputDelta] = None

    final override def backward(deltaFuture: Do[ OutputDelta]): Future[Unit] = {

      import com.thoughtworks.raii.covariant.ResourceT.resourceTMonad

      val Do(resourceFactoryTFuture) = deltaFuture

      val resourceFactoryT: ResourceT[Future, Try[OutputDelta]] = ResourceT(resourceFactoryTFuture)

      val tryTRAIIFuture: ResourceT[Future, Try[Unit]] = resourceFactoryT.map { tryDelta: Try[OutputDelta] =>
        tryDelta.map { delta =>
          synchronized {
            if (logger.isLoggable(Level.FINER)) {
              logger.log(DeltaAccumulatorIsUpdating(deltaAccumulator, delta))
            }
            deltaAccumulator |+|= Some(delta)
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

  trait Binary[OutputData, OutputDelta] {

    def apply[Data0, Delta0, Data1, Delta1](operand0: Do[ Borrowing[Tape.Aux[Data0, Delta0]]],
                                            operand1: Do[ Borrowing[Tape.Aux[Data1, Delta1]]])(
        computeForward: (Data0, Data1) => Task[(OutputData, OutputDelta => (Do[ Delta0], Do[ Delta1]))])
      : Do[Borrowing[Tape.Aux[OutputData, OutputDelta]]]

  }

  trait Unary[OutputData, OutputDelta] {

    def apply[Data, Delta](operand: Do[ Borrowing[Tape.Aux[Data, Delta]]])(
        computeForward: (Data) => Task[(OutputData, OutputDelta => Do[ Delta])])
      : Do[Borrowing[Tape.Aux[OutputData, OutputDelta]]]
  }

  object Binary {

    @inline
    def doTape[Data0, Delta0, Data1, Delta1, OutputData, OutputDelta](
        operand0: Do[ Borrowing[Tape.Aux[Data0, Delta0]]],
        operand1: Do[ Borrowing[Tape.Aux[Data1, Delta1]]])(
        computeForward: (Data0, Data1) => Task[(OutputData, OutputDelta => (Do[ Delta0], Do[ Delta1]))])(
        implicit binaryTapeTaskFactory: Binary[OutputData, OutputDelta],
        logger: Logger,
        fullName: sourcecode.FullName,
        methodName: sourcecode.Name,
        className: Caller[_]): Do[Borrowing[Tape.Aux[OutputData, OutputDelta]]] = {
      binaryTapeTaskFactory(operand0, operand1)(computeForward)
    }

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

    final class MonoidBinary[OutputData, OutputDelta: Monoid](implicit logger: Logger,
                                                              fullName: sourcecode.FullName,
                                                              methodName: sourcecode.Name,
                                                              className: Caller[_])
        extends Binary[OutputData, OutputDelta] {
      @inline
      override def apply[Data0, Delta0, Data1, Delta1](operand0: Do[ Borrowing[Tape.Aux[Data0, Delta0]]],
                                                       operand1: Do[ Borrowing[Tape.Aux[Data1, Delta1]]])(
          computeForward: (Data0, Data1) => Task[(OutputData, OutputDelta => (Do[ Delta0], Do[ Delta1]))])
        : Do[Borrowing[Tape.Aux[OutputData, OutputDelta]]] = {

        import com.thoughtworks.raii.covariant.ResourceT.resourceTParallelApplicative
        import com.thoughtworks.raii.asynchronous.Do.doParallelApplicative

        val parallelTuple =
          Applicative[Lambda[x => Do[x] @@ Parallel]]
            .tuple2(Parallel(operand0), Parallel(operand1))

        val tuple = Parallel.unwrap(parallelTuple)

        import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

        tuple.flatMap { pair =>
          val (upstream0, upstream1) = pair
          val resource: ResourceT[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]] = {
            val futureResourceT: Future[Releasable[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]]] =
              computeForward(upstream0.data, upstream1.data).get.map {
                case left @ -\/(e) =>
                  new Releasable[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]] {
                    override def release(): Future[Unit] = Future.now(())
                    override def value: Try[Borrowing[Aux[OutputData, OutputDelta]]] = Failure(e)
                  }
                case right @ \/-((forwardData, computeBackward)) =>
                  new MonoidOutput[OutputData, OutputDelta](forwardData) {
                    override def release(): Future[Unit] = {
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
        className: Caller[_]): Binary[OutputData, OutputDelta] = {
      new MonoidBinary[OutputData, OutputDelta]
    }

    final class SemigroupBinary[OutputData, OutputDelta: Semigroup](implicit logger: Logger,
                                                                    fullName: sourcecode.FullName,
                                                                    methodName: sourcecode.Name,
                                                                    className: Caller[_])
        extends Binary[OutputData, OutputDelta] {
      @inline
      def apply[Data0, Delta0, Data1, Delta1](operand0: Do[ Borrowing[Tape.Aux[Data0, Delta0]]],
                                              operand1: Do[ Borrowing[Tape.Aux[Data1, Delta1]]])(
          computeForward: (Data0, Data1) => Task[(OutputData, (OutputDelta) => (Do[ Delta0], Do[ Delta1]))])
        : Do[Borrowing[Tape.Aux[OutputData, OutputDelta]]] = {
        import com.thoughtworks.raii.covariant.ResourceT.resourceTParallelApplicative
        import com.thoughtworks.raii.asynchronous.Do.doParallelApplicative

        val parallelTuple =
          Applicative[Lambda[x => Do[x] @@ Parallel]]
            .tuple2(Parallel(operand0), Parallel(operand1))

        val tuple = Parallel.unwrap(parallelTuple)

        import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances

        tuple.flatMap { pair =>
          val (upstream0, upstream1) = pair
          val resource: ResourceT[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]] = {
            val futureResourceT: Future[Releasable[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]]] =
              computeForward(upstream0.data, upstream1.data).get.map {
                case left @ -\/(e) =>
                  new Releasable[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]] {
                    override def release(): Future[Unit] = Future.now(())
                    override def value: Try[Borrowing[Aux[OutputData, OutputDelta]]] = Failure(e)
                  }
                case right @ \/-((forwardData, computeBackward)) =>
                  new SemigroupOutput[OutputData, OutputDelta](forwardData) {
                    override def release(): Future[Unit] = {
                      deltaAccumulator match {
                        case Some(deltaAcc) =>
                          val (upstream0DeltaFuture, upstream1DeltaFuture) = computeBackward(deltaAcc)
                          Parallel.unwrap {
                            Future.futureParallelApplicativeInstance.apply2(
                              Parallel(upstream0.backward(upstream0DeltaFuture)),
                              Parallel(upstream1.backward(upstream1DeltaFuture))
                            ) { (_: Unit, _: Unit) =>
                              ()
                            }
                          }
                        case None => Future.now(())
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
    implicit def semigroupBinaryTapeTaskFactory[OutputData, OutputDelta: Semigroup](
        implicit logger: Logger,
        fullName: sourcecode.FullName,
        methodName: sourcecode.Name,
        className: Caller[_]): Binary[OutputData, OutputDelta] = {
      new SemigroupBinary[OutputData, OutputDelta]
    }
  }

  object Unary {

    @inline
    def doTape[Data, Delta, OutputData, OutputDelta](operand: Do[ Borrowing[Tape.Aux[Data, Delta]]])(
        computeForward: (Data) => Task[(OutputData, OutputDelta => Do[ Delta])])(
        implicit unaryTapeTaskFactory: Unary[OutputData, OutputDelta],
        logger: Logger,
        fullName: sourcecode.FullName,
        methodName: sourcecode.Name,
        className: Caller[_]): Do[Borrowing[Tape.Aux[OutputData, OutputDelta]]] = {
      unaryTapeTaskFactory(operand)(computeForward)
    }

    final class MonoidUnary[OutputData, OutputDelta: Monoid](implicit logger: Logger,
                                                             fullName: sourcecode.FullName,
                                                             methodName: sourcecode.Name,
                                                             className: Caller[_])
        extends Unary[OutputData, OutputDelta] {
      @inline
      override def apply[Data, Delta](operand: Do[ Borrowing[Tape.Aux[Data, Delta]]])(
          computeForward: Data => Task[(OutputData, OutputDelta => Do[ Delta])])
        : Do[Borrowing[Tape.Aux[OutputData, OutputDelta]]] = {
        import com.thoughtworks.raii.covariant.ResourceT._
        import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances
        operand.flatMap {
          case (upstream) =>
            val resource: ResourceT[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]] = {
              val futureResourceT: Future[Releasable[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]]] =
                computeForward(upstream.data).get.map {
                  case left @ -\/(e) =>
                    new Releasable[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]] {
                      override def release(): Future[Unit] = Future.now(())
                      override def value: Try[Borrowing[Aux[OutputData, OutputDelta]]] = Failure(e)
                    }
                  case right @ \/-((forwardData, computeBackward)) =>
                    new MonoidOutput[OutputData, OutputDelta](forwardData) {
                      override def release(): Future[Unit] = {
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
        className: Caller[_]): Unary[OutputData, OutputDelta] = {
      new MonoidUnary[OutputData, OutputDelta]
    }

    final class SemigroupUnary[OutputData, OutputDelta: Semigroup](implicit logger: Logger,
                                                                   fullName: sourcecode.FullName,
                                                                   methodName: sourcecode.Name,
                                                                   className: Caller[_])
        extends Unary[OutputData, OutputDelta] {
      @inline
      override def apply[Data, Delta](operand: Do[ Borrowing[Tape.Aux[Data, Delta]]])(
          computeForward: Data => Task[(OutputData, OutputDelta => Do[ Delta])])
        : Do[Borrowing[Tape.Aux[OutputData, OutputDelta]]] = {
        import com.thoughtworks.raii.covariant.ResourceT.resourceTMonadError
        import com.thoughtworks.raii.asynchronous.Do.doMonadErrorInstances
        operand.flatMap {
          case (upstream) =>
            val resource: ResourceT[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]] = {
              val futureResourceT: Future[Releasable[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]]] =
                computeForward(upstream.data).get.map {
                  case left @ -\/(e) =>
                    new Releasable[Future, Try[Borrowing[Tape.Aux[OutputData, OutputDelta]]]] {
                      override def release(): Future[Unit] = Future.now(())
                      override def value: Try[Borrowing[Aux[OutputData, OutputDelta]]] = Failure(e)
                    }
                  case right @ \/-((forwardData, computeBackward)) =>
                    new SemigroupOutput[OutputData, OutputDelta](forwardData) {
                      override def release(): Future[Unit] = {
                        deltaAccumulator match {
                          case Some(deltaAcc) => upstream.backward(computeBackward(deltaAcc))
                          case None => Future.now(())
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
    implicit def semigroupUnaryTapeTaskFactory[OutputData, OutputDelta: Semigroup](
        implicit logger: Logger,
        fullName: sourcecode.FullName,
        methodName: sourcecode.Name,
        className: Caller[_]): Unary[OutputData, OutputDelta] = {
      new SemigroupUnary[OutputData, OutputDelta]
    }
  }

}
