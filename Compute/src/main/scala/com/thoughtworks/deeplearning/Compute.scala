package com.thoughtworks.deeplearning

import java.util.concurrent.ExecutorService

import com.thoughtworks.deeplearning.Tape.{Aux, Literal}
import com.thoughtworks.raii.ResourceFactoryT
import com.thoughtworks.raii.ResourceFactoryT.ReleasableT

import scalaz.{-\/, Applicative, Bind, EitherT, Functor, Id, Monoid, Nondeterminism, \/, \/-}
import scalaz.concurrent.{Future, Task}
import com.thoughtworks.raii.EitherTNondeterminism._
import com.thoughtworks.raii.Shared._

import scalaz.syntax.all._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object Compute {

  def managed[A <: AutoCloseable](task: Task[A]): Compute[A] = {
    new Compute[A]({ () =>
      task.get.map { either =>
        new ReleasableT[Future, Throwable \/ A] {
          override def value: Throwable \/ A = either
          override def release(): Future[Unit] = {
            either match {
              case \/-(closeable) =>
                Future.delay(closeable.close())
              case -\/(e) =>
                Future.now(())
            }
          }
        }
      }
    })
  }

  def managed[A <: AutoCloseable](future: Future[A]): Compute[A] = {
    managed(new Task(future.map(\/-(_))))
  }

  def managed[A <: AutoCloseable](a: => A): Compute[A] = {
    managed(Task.delay(a))
  }

  def unmanaged[A](task: Task[A]): Compute[A] = {
    new Compute[A]({ () =>
      task.get.map { either =>
        new ReleasableT[Future, Throwable \/ A] {
          override def value: Throwable \/ A = either
          override def release(): Future[Unit] = Future.now(())
        }
      }
    })
  }

  def unmanaged[A](future: Future[A]): Compute[A] = {
    unmanaged(new Task(future.map(\/-(_))))
  }

  def unmanaged[A](a: => A): Compute[A] = {
    unmanaged(Task.delay(a))
  }
  
  def delay[A](a: => A): Compute[A] = {
    unmanaged(a)
  }

  /** Create a [[Compute]] that will evaluate `a` using the given `ExecutorService`. */
  def apply[A](a: => A)(implicit executorService: ExecutorService): Compute[A] = {
    new Compute[A]({ () =>
      Task(a).get.map { either =>
        new ReleasableT[Future, Throwable \/ A] {
          override def value: Throwable \/ A = either
          override def release(): Future[Unit] = Future.now(())
        }
      }
    })
  }

  @inline
  def binary[Data0, Delta0, Data1, Delta1, OutputData, OutputDelta](operand0: Compute[Tape.Aux[Data0, Delta0]],
                                                                    operand1: Compute[Tape.Aux[Data1, Delta1]])(
      computeForward: (Data0, Data1) => Task[(OutputData, OutputDelta => (Future[Delta0], Future[Delta1]))])(
      implicit binaryComputeFactory: BinaryComputeFactory[OutputData, OutputDelta])
    : Compute[Tape.Aux[OutputData, OutputDelta]] = {
    binaryComputeFactory(operand0, operand1)(computeForward)
  }

  @inline
  def unary[Data, Delta, OutputData, OutputDelta](operand: Compute[Tape.Aux[Data, Delta]])(
      computeForward: (Data) => Task[(OutputData, OutputDelta => Future[Delta])])(
      implicit unaryComputeFactory: UnaryComputeFactory[OutputData, OutputDelta])
    : Compute[Tape.Aux[OutputData, OutputDelta]] = {
    unaryComputeFactory(operand)(computeForward)
  }

  private[deeplearning] type FutureResourceFactory[Result] = ResourceFactoryT[Future, Result]

  private[deeplearning] type Compute[Result] = EitherT[FutureResourceFactory, Throwable, Result]

  private trait Output[OutputData, OutputDelta]
      extends ReleasableT[Future, Throwable \/ Tape.Aux[OutputData, OutputDelta]] { this: Tape =>
    override final type Delta = OutputDelta
    override final type Data = OutputData
    final override def value: \/-[this.type] = \/-(this)
  }

  private final class UntrainableOutput[OutputData, OutputDelta](override val data: OutputData)
      extends Tape
      with Output[OutputData, OutputDelta] {
    override def release(): Future[Unit] = Future.now(())

    override def backward(delta: Future[Delta]): Future[Unit] = Future.now(())
  }

  private abstract class output[OutputData, OutputDelta: Monoid](override val data: OutputData)
      extends Tape
      with Output[OutputData, OutputDelta] {

    @volatile
    protected var deltaAccumulator: OutputDelta = mzero[OutputDelta]

    final override def backward(deltaFuture: Future[OutputDelta]): Future[Unit] = {
      deltaFuture.map { delta =>
        synchronized {
          deltaAccumulator |+|= delta
        }
      }
    }

  }

  private abstract class Releasable[OutputData, OutputDelta](
      override val value: \/[Throwable, Aux[OutputData, OutputDelta]])
      extends ReleasableT[Future, Throwable \/ Tape.Aux[OutputData, OutputDelta]] {
    override def release(): Future[Unit] = Future.now(())
  }

  trait BinaryComputeFactory[OutputData, OutputDelta] {

    def apply[Data0, Delta0, Data1, Delta1](operand0: Compute[Tape.Aux[Data0, Delta0]],
                                            operand1: Compute[Tape.Aux[Data1, Delta1]])(
        computeForward: (Data0, Data1) => Task[(OutputData, OutputDelta => (Future[Delta0], Future[Delta1]))])
      : Compute[Tape.Aux[OutputData, OutputDelta]]

  }

  trait UnaryComputeFactory[OutputData, OutputDelta] {

    def apply[Data, Delta](operand: Compute[Tape.Aux[Data, Delta]])(
        computeForward: (Data) => Task[(OutputData, OutputDelta => (Future[Delta]))])
      : Compute[Tape.Aux[OutputData, OutputDelta]]
  }

  object BinaryComputeFactory {
    final class MonoidBinaryComputeFactory[OutputData, OutputDelta: Monoid]
        extends BinaryComputeFactory[OutputData, OutputDelta] {
      override def apply[Data0, Delta0, Data1, Delta1](operand0: Compute[Tape.Aux[Data0, Delta0]],
                                                       operand1: Compute[Tape.Aux[Data1, Delta1]])(
          computeForward: (Data0, Data1) => Task[(OutputData, OutputDelta => (Future[Delta0], Future[Delta1]))])
        : Compute[Tape.Aux[OutputData, OutputDelta]] = {
        Nondeterminism[Compute].both(operand0, operand1).flatMap {
          case (upstream0, upstream1) =>
            val resource: FutureResourceFactory[Throwable \/ Tape.Aux[OutputData, OutputDelta]] = { () =>
              computeForward(upstream0.data, upstream1.data).get.map {
                case left @ -\/(_) =>
                  new Releasable[OutputData, OutputDelta](left) {}
                case right @ \/-((forwardData, computeBackward)) =>
                  new output[OutputData, OutputDelta](forwardData) {
                    override def release(): Future[Unit] = {
                      val (upstream0DeltaFuture, upstream1DeltaFuture) = computeBackward(deltaAccumulator)
                      Future.futureInstance.mapBoth(
                        upstream0.backward(upstream0DeltaFuture),
                        upstream1.backward(upstream1DeltaFuture)
                      ) { (_: Unit, _: Unit) =>
                        ()
                      }
                    }
                  }
              }
            }
            new Compute(resource.shared)
        }
      }
    }

    @inline
    implicit def monoidBinaryComputeFactory[OutputData, OutputDelta: Monoid]
      : BinaryComputeFactory[OutputData, OutputDelta] = {
      new MonoidBinaryComputeFactory[OutputData, OutputDelta]
    }
  }

  object UnaryComputeFactory {
    final class MonoidUnaryComputeFactory[OutputData, OutputDelta: Monoid]
        extends UnaryComputeFactory[OutputData, OutputDelta] {
      override def apply[Data, Delta](operand: Compute[Tape.Aux[Data, Delta]])(
          computeForward: (Data) => Task[(OutputData, OutputDelta => Future[Delta])])
        : Compute[Tape.Aux[OutputData, OutputDelta]] = {
        operand.flatMap {
          case (upstream) =>
            val resource: FutureResourceFactory[Throwable \/ Tape.Aux[OutputData, OutputDelta]] = { () =>
              computeForward(upstream.data).get.map {
                case left @ -\/(_) =>
                  new Releasable[OutputData, OutputDelta](left) {}
                case right @ \/-((forwardData, computeBackward)) =>
                  upstream.workaround10251 match {
                    case trainable: Tape =>
                      new output[OutputData, OutputDelta](forwardData) {
                        override def release(): Future[Unit] = {
                          trainable.backward(computeBackward(deltaAccumulator))
                        }
                      }
                    case untrainable: Tape =>
                      new UntrainableOutput[OutputData, OutputDelta](forwardData)
                  }
              }
            }
            new Compute(resource.shared)
        }
      }
    }

    @inline
    implicit def monoidUnaryComputeFactory[OutputData, OutputDelta: Monoid]
      : UnaryComputeFactory[OutputData, OutputDelta] = {
      new MonoidUnaryComputeFactory[OutputData, OutputDelta]
    }
  }

  implicit def floatToCompute(value: Float): Compute[Tape.Aux[Float, Float]] = {
    Applicative[Compute].point(Literal(value))
  }

}
