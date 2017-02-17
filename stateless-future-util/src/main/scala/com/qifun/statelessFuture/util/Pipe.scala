package com.qifun.statelessFuture.util

import com.qifun.statelessFuture.Awaitable
import scala.util.control.TailCalls._
import scala.util.control.Exception.Catcher
import java.util.concurrent.atomic.AtomicReference
import com.qifun.statelessFuture.AwaitableFactory
import scala.collection.LinearSeq

object Pipe {

  // Don't extend AnyVal because of https://issues.scala-lang.org/browse/SI-7449
  private[Pipe] final case class TransitionFunction[Event](transitionFunction: Event => TailRec[TransitionFunction[Event]]) /* extends AnyVal */

  private[Pipe] sealed trait State[Event]

  private[Pipe] final case class WaitingForEvent[Event](transitionFunction: Event => TailRec[TransitionFunction[Event]]) extends State[Event]

  private[Pipe] final case class PendingEvent[Event](rest: List[Event]) extends State[Event]

  final class Writer[Event] private[Pipe] (state: State[Event]) extends AtomicReference[State[Event]](state) {
    private def consumeRest(newTransitionFunction: Event => TailRec[TransitionFunction[Event]]): Unit = {
      get() match {
        case oldState @ PendingEvent(Nil) => {
          if (!compareAndSet(oldState, WaitingForEvent(newTransitionFunction))) {
            consumeRest(newTransitionFunction)
          }
        }
        case oldState @ PendingEvent(rest) => {
          if (compareAndSet(oldState, PendingEvent(Nil))) {
            consumeRest(rest.foldRight(newTransitionFunction) { (event, transitionFunction) =>
              transitionFunction(event).result.transitionFunction
            })
          } else {
            consumeRest(newTransitionFunction)
          }

        }
        case _ => {
          throw new IllegalStateException
        }
      }
    }

    @scala.annotation.tailrec
    final def write(event: Event): Unit = {
      get() match {
        case oldState @ PendingEvent(rest) => {
          if (!compareAndSet(oldState, PendingEvent(event :: rest))) {
            write(event)
          }
        }
        case oldState @ WaitingForEvent(transitionFunction) => {
          if (compareAndSet(oldState, PendingEvent(Nil))) {
            consumeRest(transitionFunction(event).result.transitionFunction)
          } else {
            write(event)
          }
        }
      }
    }

  }

  final class Reader[Event]() extends Awaitable.Stateless[Event, TransitionFunction[Event]] {
    override final def onComplete(handler: Event => TailRec[TransitionFunction[Event]])(implicit catcher: Catcher[TailRec[TransitionFunction[Event]]]): TailRec[TransitionFunction[Event]] = {
      done(new TransitionFunction[Event](handler))
    }

  }

}

final case class Pipe[Event]() {

  import Pipe.TransitionFunction

  def read() = new Pipe.Reader[Event]()

  type Future[AwaitResult] = Awaitable.Stateless[AwaitResult, TransitionFunction[Event]]

  object Future extends AwaitableFactory[TransitionFunction[Event]]

  final def start(readingFuture: Pipe[Event]#Future[Nothing]): Pipe.Writer[Event] = {
    def catcher: Catcher[TransitionFunction[Event]] = PartialFunction.empty // 此处不捕获异常，统一在 write 处捕获
    val writer = (for (_ <- readingFuture) { ??? })(catcher)
    new Pipe.Writer[Event](Pipe.WaitingForEvent(writer.transitionFunction))
  }

  private type FutureSeq[A] = AwaitableSeq[A, TransitionFunction[Event]]

  final def futureSeq[A](underlying: LinearSeq[A]) = new FutureSeq[A](underlying)

  final def futureSeq[A](underlying: TraversableOnce[A]) = new FutureSeq[A](Generator.GeneratorSeq(underlying))

  
}