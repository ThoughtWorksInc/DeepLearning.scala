package com.thoughtworks.deeplearning

import com.thoughtworks.raii.ResourceFactoryT
import com.thoughtworks.raii.ResourceFactoryT.ReleasableT

import scalaz.{-\/, EitherT, Monoid, Nondeterminism, \/, \/-}
import scalaz.concurrent.{Future, Task}
import com.thoughtworks.raii.EitherTNondeterminism._
import com.thoughtworks.raii.Shared._

import scalaz.syntax.all._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object Compute {

  def binary[Data0, Delta0, Data1, Delta1, OutputData, OutputDelta](operand0: Compute[Tape.Aux[Data0, Delta0]],
                                                                    operand1: Compute[Tape.Aux[Data1, Delta1]])(
      computeForward: (Data0, Data1) => Task[(OutputData, OutputDelta => (Future[Delta0], Future[Delta1]))])(
      implicit factory: ComputeFactory[OutputData, OutputDelta]): Compute[Tape.Aux[OutputData, OutputDelta]] = {
    factory.binary(operand0, operand1)(computeForward)
  }

  private[deeplearning] type FutureResourceFactory[A] = ResourceFactoryT[Future, A]

  private[deeplearning] type Compute[A] = EitherT[FutureResourceFactory, Throwable, A]

  trait ComputeFactory[OutputData, OutputDelta] {

    def binary[Data0, Delta0, Data1, Delta1](operand0: Compute[Tape.Aux[Data0, Delta0]],
                                             operand1: Compute[Tape.Aux[Data1, Delta1]])(
        computeForward: (Data0, Data1) => Task[(OutputData, OutputDelta => (Future[Delta0], Future[Delta1]))])
      : Compute[Tape.Aux[OutputData, OutputDelta]]

  }

  object ComputeFactory {
    final class MonoidComputeFactory[OutputData, OutputDelta: Monoid] extends ComputeFactory[OutputData, OutputDelta] {
      override def binary[Data0, Delta0, Data1, Delta1](operand0: Compute[Tape.Aux[Data0, Delta0]],
                                                        operand1: Compute[Tape.Aux[Data1, Delta1]])(
          computeForward: (Data0, Data1) => Task[(OutputData, OutputDelta => (Future[Delta0], Future[Delta1]))])
        : Compute[Tape.Aux[OutputData, OutputDelta]] = {
        Nondeterminism[Compute].both(operand0, operand1).flatMap {
          case (upstream0, upstream1) =>
            val resource: FutureResourceFactory[Throwable \/ Tape.Aux[OutputData, OutputDelta]] = { () =>
              computeForward(upstream0.data, upstream1.data).get.map {
                case left @ -\/(_) =>
                  new ReleasableT[Future, Throwable \/ Tape.Aux[OutputData, OutputDelta]] {
                    override def value: -\/[Throwable] = left

                    override def release(): Future[Unit] = Future.now(())
                  }
                case right @ \/-((forwardData, computeBackward)) =>
                  trait Output extends ReleasableT[Future, Throwable \/ Tape.Aux[OutputData, OutputDelta]] {
                    this: Tape =>
                    override final type Delta = OutputDelta
                    override final type Data = OutputData
                    final override def data: Data = forwardData
                    final override def value: \/-[this.type] = \/-(this)
                  }
                  final class UntrainableOutput extends Tape.Untrainable with Output {
                    override def release(): Future[Unit] = Future.now(())
                  }
                  trait TrainableOutput extends Tape.Trainable with Output {
                    @volatile
                    protected var deltaAccumulator: OutputDelta = mzero[OutputDelta]
                    final override def backward(delta: OutputDelta): Future[Unit] = Future.delay {
                      synchronized {
                        deltaAccumulator |+|= delta
                      }
                    }
                  }
                  upstream0 match {
                    case trainable0: Tape.Trainable =>
                      upstream1 match {
                        case trainable1: Tape.Trainable =>
                          new TrainableOutput {
                            override def release(): Future[Unit] = {
                              val (upstream0Delta, upstream1Delta) = computeBackward(deltaAccumulator)
                              Future.futureInstance.mapBoth(
                                upstream0Delta.flatMap(trainable0.backward),
                                upstream1Delta.flatMap(trainable1.backward)
                              ) { (_: Unit, _: Unit) =>
                                ()
                              }
                            }
                          }
                        case untrainable1: Tape.Untrainable =>
                          new TrainableOutput {
                            override def release(): Future[Unit] = {
                              val (upstream0Delta, upstream1Delta) = computeBackward(deltaAccumulator)
                              upstream0Delta.flatMap(trainable0.backward)
                            }
                          }
                      }
                    case untrainable0: Tape.Untrainable =>
                      upstream1 match {
                        case trainable1: Tape.Trainable =>
                          new TrainableOutput {
                            override def release(): Future[Unit] = {
                              val (upstream0Delta, upstream1Delta) = computeBackward(deltaAccumulator)
                              upstream1Delta.flatMap(trainable1.backward)
                            }
                          }
                        case untrainable1: Tape.Untrainable =>
                          new UntrainableOutput
                      }
                  }
              }
            }
            new Compute(resource.shared)
        }
      }
    }

    implicit def monoidComputeFactory[OutputData, OutputDelta: Monoid]
      : MonoidComputeFactory[OutputData, OutputDelta] = {
      new MonoidComputeFactory[OutputData, OutputDelta]
    }
  }
}
