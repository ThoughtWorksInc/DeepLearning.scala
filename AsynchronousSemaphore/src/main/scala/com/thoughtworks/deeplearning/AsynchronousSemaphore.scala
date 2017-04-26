package com.thoughtworks.deeplearning

import java.util.concurrent.atomic.AtomicReference

import scala.annotation.tailrec
import scala.collection.immutable.Queue
import scalaz.{ContT, Trampoline}
import scalaz.Free.Trampoline

object AsynchronousSemaphore {
  sealed trait State
  final case class Available(restNumberOfPermits: Int) extends State
  final case class Unavailable(waiters: Queue[Unit => Trampoline[Unit]]) extends State

  @inline
  def apply(numberOfPermits: Int): AsynchronousSemaphore = {
    numberOfPermits.ensuring(_ > 0)
    new AtomicReference[State](Available(numberOfPermits)) with AsynchronousSemaphore {
      override protected def state: AtomicReference[State] = this
    }
  }
}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait AsynchronousSemaphore {
  import AsynchronousSemaphore._
  protected def state: AtomicReference[State]

  final def acquire(): ContT[Trampoline, Unit, Unit] = {
    ContT[Trampoline, Unit, Unit]({ waiter: (Unit => Trampoline[Unit]) =>
      @tailrec
      def retry(): Trampoline[Unit] = {
        state.get() match {
          case oldState @ Available(1) =>
            if (state.compareAndSet(oldState, Unavailable(Queue.empty))) {
              waiter(())
            } else {
              retry()
            }
          case oldState @ Available(restNumberOfPermits) if restNumberOfPermits > 1 =>
            if (state.compareAndSet(oldState, Available(restNumberOfPermits - 1))) { // TODO
              waiter(())
            } else {
              retry()
            }
          case oldState @ Unavailable(waiters) =>
            if (state.compareAndSet(oldState, Unavailable(waiters.enqueue(waiter)))) {
              Trampoline.done(())
            } else {
              retry()
            }
        }
      }
      retry()
    })
  }

  @tailrec
  final def release(): Trampoline[Unit] = {
    state.get() match {
      case oldState @ Unavailable(waiters) =>
        val (head, tail) = waiters.dequeue
        if (state.compareAndSet(oldState, Unavailable(tail))) {
          head(())
        } else {
          release()
        }
      case oldState @ Available(restNumberOfPermits) =>
        if (state.compareAndSet(oldState, Available(restNumberOfPermits + 1))) {
          Trampoline.done(())
        } else {
          release()
        }
    }
  }
}
