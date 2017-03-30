package com.thoughtworks.deeplearning

import java.util.concurrent.atomic.AtomicReference

import com.thoughtworks.deeplearning.CumulativeTape.MonoidTape
import com.thoughtworks.deeplearning.Layer.Tape
import shapeless.the

import scala.util.{Failure, Success, Try}
import scala.util.control.Exception.Catcher
import scala.util.control.TailCalls._
import cats.Monoid
import com.thoughtworks.deeplearning.Layer.Tape.Aux
import com.thoughtworks.future.Continuation.Task
import com.thoughtworks.future.Future
import com.thoughtworks.future.Future.Zip

import scalaz.syntax.zip._
import com.thoughtworks.future.scalaz.TaskInstance.scalazTaskInstance
import com.thoughtworks.future.sde.task

import scala.annotation.tailrec
import scala.collection.immutable.Queue
import scala.util.control.TailCalls
import scalaz.{Functor, Monad}

/**
  * A namespace of common operators for Float layers.
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableFloat {

  object Tapes {

    private[Tapes] trait FloatMonoidTape extends MonoidTape { this: Tape =>

      override final type Data = Float

      override final type Delta = Float

      override protected final def monoid: Monoid[Float] = cats.instances.float.catsKernelStdGroupForFloat
    }
    trait Plus
//
//    trait Plus extends FloatMonoidTape with BinaryTape { this: Tape =>
//      override final type Upstream0 = Tape.Aux[Float, Float]
//      override final type Upstream1 = Tape.Aux[Float, Float]
//      override final val value: Float = upstream0.value + upstream1.value
//      override protected final def upstreamDelta(outputDelta: Delta) = {
//        val upstream0Delta = future { outputDelta }
//        val upstream1Delta = future { outputDelta }
//        (upstream0Delta, upstream1Delta)
//      }
//
//    }
  }

  import Tapes._

  private type FloatCache = Cache[Tape.Aux[Float, Float]]

  trait ReferenceCounter[A] {
    def get: A
    def release(): Task[Unit]
    def retain(): Unit
  }

  trait Cache[AwaitResult <: Tape] {
    def value: Option[Try[ReferenceCounter[AwaitResult]]]
    def onComplete(handler: Try[ReferenceCounter[AwaitResult]] => TailRec[Unit]): TailRec[Unit]
//    def release(): Task[Unit]
//    def retain(): Unit
  }

//  object CacheInstance extends scalaz.Zip[Cache.Aux] {
//    override def zipWith[A, B, C](fa: => Cache.Aux[A], fb: => Cache.Aux[B])(f: (A, B) => C)(
//        implicit F: Functor[Cache.Aux]): Cache.Aux[C] = {
//      ???
//    }
//    override def zip[A, B](a: => Cache.Aux[A], b: => Cache.Aux[B]): Cache.Aux[(A, B)] = {
//      new Cache {
//        type AwaitResult = (A, B)
//
//        override def value: Option[Try[ReferenceCounter[(A, B)]]] = ???
//
//        override def onComplete(handler: (Try[ReferenceCounter[(A, B)]]) => TailRec[Unit]): TailRec[Unit] = ???
//      }
//    }
//  }

//  type CacheMonad = Monad[Cache]

  def fzipWith[Upstream0 <: Tape, Upstream1 <: Tape, Output <: Tape](
      operand0: Cache[Upstream0],
      operand1: Cache[Upstream1])(f: (Upstream0, Upstream1) => Output): Cache[Output] = {
    new AtomicReference[State[Upstream0, Upstream1, Output]](GotNeither(Queue.empty)) with Cache[Output] {

      protected def state: AtomicReference[State[Upstream0, Upstream1, Output]] = this
//
//      private def dispatch[AwaitResult](handlers: Queue[Try[AwaitResult] => TailRec[Unit]],
//                                        value: Try[AwaitResult]): TailRec[Unit] = {
//        if (handlers.isEmpty) {
//          done(())
//        } else {
//          val (head, tail) = handlers.dequeue
//          head(value).flatMap { _ =>
//            dispatch(tail, value)
//          }
//        }
//      }
//
//      @tailrec
//      private[deeplearning] def release(): Task[Unit] = {
//        state.get match {
//          case oldState @ GotOutput(output, counter) =>
//            if (counter == 1) {
//              if (state.compareAndSet(oldState, GotNeither(Queue.empty))) {
//                output match {
//                  case Success(outputValue) =>
//                    outputValue.close()
//                  case Failure(e) =>
//                    task(())
//                }
//                operand0.release().flatMap { _ =>
//                  operand1.release()
//                }
//              } else {
//                release()
//              }
//            } else {
//              if (state.compareAndSet(oldState, GotOutput(output, counter - 1))) {
//                task(())
//              } else {
//                release()
//              }
//            }
//
//        }
//      }
//      @tailrec
//      private def tryCompleteOutput(output: Try[Tape.Aux[OutputData, OutputDelta]]): TailRec[Unit] = {
//        state.get match {
//          case oldState @ GotBoth(handlers) => {
//            if (state.compareAndSet(oldState, GotOutput(output, 1))) {
//              // TODO: dispatch
//              dispatch(handlers, output).flatMap { _: Unit =>
//                release().onComplete { _ =>
//                  done(())
//                }
//              }
//            } else {
//              tryCompleteOutput(output)
//            }
//          }
//          case GotOutput(_, _) =>
//            throw new IllegalStateException("Cannot complete with an output more than once!")
//          case GotA(_, _) | GotB(_, _) | GotNeither(_) =>
//            throw new IllegalStateException("Cannot complete with an output before got both operands!")
//        }
//      }
//
//      /**
//        *
//        * @param value
//        * @return
//        */
      private def tryCompleteA(value: Try[ReferenceCounter[Upstream0]]): TailRec[Unit] = {
//        state.get match {
//          case oldState @ GotNeither(handlers) => {
//            if (state.compareAndSet(oldState, GotA(value, handlers))) {
//              // TODO: Retain
//              done(())
//            } else {
//              tryCompleteA(value)
//            }
//          }
//          case oldState @ GotB(tryB, handlers) => {
//            if (state.compareAndSet(oldState, GotBoth(handlers))) {
//              value match {
//                case Success(a) =>
//                  tryB match {
//                    case Success(b) =>
//                      val output = Try(f(this, a, b))
//                      tailcall(tryCompleteOutput(output))
//                    case Failure(e) =>
//                      ???
//                  }
//                case Failure(e) =>
//                  ???
//              }
//            } else {
//              tryCompleteA(value)
//            }
//          }
//          case GotA(_, _) | GotBoth(_) | GotOutput(_, _) =>
//            throw new IllegalStateException("Cannot complete with the operand0 more than once!")
//        }
        ???
      }

      private def tryCompleteB(value: Try[ReferenceCounter[Upstream1]]): TailRec[Unit] = ???

//
//      override def onComplete(handler: (Try[Aux[OutputData, OutputDelta]]) => TailRec[Unit]): TailRec[Unit] = {
//        state.get match {
//          case oldState @ GotNeither(handlers) => {
//            if (state.compareAndSet(oldState, GotNeither(handlers.enqueue(handler)))) {
//              if (handlers.isEmpty) {
//                // TODO: start operand0 and operand1
//                ???
//              } else {
//                done(())
//              }
//            } else {
//              onComplete(handler)
//            }
//          }
//          case oldState @ GotA(a, handlers) => {
//            if (state.compareAndSet(oldState, GotA(a, handlers.enqueue(handler)))) {
//              done(())
//            } else {
//              onComplete(handler)
//            }
//          }
//          case oldState @ GotB(b, handlers) => {
//            if (state.compareAndSet(oldState, GotB(b, handlers.enqueue(handler)))) {
//              done(())
//            } else {
//              onComplete(handler)
//            }
//          }
//          case oldState @ GotBoth(handlers) => {
//            if (state.compareAndSet(oldState, GotBoth(handlers.enqueue(handler)))) {
//              done(())
//            } else {
//              onComplete(handler)
//            }
//          }
//          case oldState @ GotOutput(output, counter) => {
//            ???
//          }
//        }
//      }
      override def value: Option[Try[ReferenceCounter[Output]]] = ???

      override def onComplete(handler: (Try[ReferenceCounter[Output]]) => TailRec[Unit]): TailRec[Unit] = {
        state.get match {
          case oldState @ GotNeither(handlers) => {
            if (state.compareAndSet(oldState, GotNeither(handlers.enqueue(handler)))) {
              if (handlers.isEmpty) {
                operand0.onComplete(tryCompleteA).flatMap { _: Unit =>
                  operand1.onComplete(tryCompleteB)
                }
              } else {
                done(())
              }
            } else {
              onComplete(handler)
            }
          }
          case oldState @ GotA(a, handlers) => {
            if (state.compareAndSet(oldState, GotA(a, handlers.enqueue(handler)))) {
              done(())
            } else {
              onComplete(handler)
            }
          }
          case oldState @ GotB(b, handlers) => {
            if (state.compareAndSet(oldState, GotB(b, handlers.enqueue(handler)))) {
              done(())
            } else {
              onComplete(handler)
            }
          }
          case oldState @ GotBoth(handlers) => {
            if (state.compareAndSet(oldState, GotBoth(handlers.enqueue(handler)))) {
              done(())
            } else {
              onComplete(handler)
            }
          }
          case oldState @ GotOutput(output, counter) => {
            ???
          }
          case GotException(_) =>
            ???
        }
      }
    }
  }

  private sealed trait State[Upstream0 <: Tape, Upstream1 <: Tape, Output <: Tape]

  private final case class GotNeither[Upstream0 <: Tape, Upstream1 <: Tape, Output <: Tape](
      handlers: Queue[Try[ReferenceCounter[Output]] => TailRec[Unit]])
      extends State[Upstream0, Upstream1, Output]

  private final case class GotA[Upstream0 <: Tape, Upstream1 <: Tape, Output <: Tape](
      a: Try[ReferenceCounter[Upstream0]],
      handlers: Queue[Try[ReferenceCounter[Output]] => TailRec[Unit]])
      extends State[Upstream0, Upstream1, Output]

  private final case class GotB[Upstream0 <: Tape, Upstream1 <: Tape, Output <: Tape](
      b: Try[ReferenceCounter[Upstream1]],
      handlers: Queue[Try[ReferenceCounter[Output]] => TailRec[Unit]])
      extends State[Upstream0, Upstream1, Output]

  private final case class GotBoth[Upstream0 <: Tape, Upstream1 <: Tape, Output <: Tape](
      handlers: Queue[Try[ReferenceCounter[Output]] => TailRec[Unit]])
      extends State[Upstream0, Upstream1, Output]

  private final case class GotOutput[Upstream0 <: Tape, Upstream1 <: Tape, Output <: Tape](output: Output,
                                                                                           counter: Int)
      extends State[Upstream0, Upstream1, Output]

  private final case class GotException[Upstream0 <: Tape, Upstream1 <: Tape, Output <: Tape](throwable: Throwable)
      extends State[Upstream0, Upstream1, Output]

  implicit final class FloatTapeOps(operand0: Cache[Tape.Aux[Float, Float]]) {
    def +(operand1: Cache[Tape.Aux[Float, Float]]): FloatCache = {
//      operand1.flatMap { t1 =>
//        //val t2 = t1.duplicate()
//        /*
//      operand1.retain()
//        operand0.map { t0 =>
//          try {
//          if (t0.value + operand1.value.get.value > 1.0) {
//            new Tape
//          } else {
//            new Tape
//          }
//          }finally {
//            operand1.release().!
//          }
//        }
//
//         */
//        ???
//      }

//      def

      fzipWith(operand0, operand1) { (upstream0: Tape.Aux[Float, Float], upstream1: Tape.Aux[Float, Float]) =>
//          upstream0.retain
//          operand0.retain()
//          operand1.retain()
//          new PlusTape(upstream0, upstream1)
//        cache.newTape[Plus](upstream0, upstream1)

        ???
      }

      //CumulativeTape.makeFuture[Plus](operand0, operand1)
    }

//    class PlusTape(upstream0: Tape.Aux[Float, Float], upstream1: Tape.Aux[Float, Float]) {
//      operand0.retain()
//      operand1.retain()
//
//    }

  }
  // implicit helpers, ops, ...
}

/*

val a: Future[...] = b + c

val d = a dot w


def train(f: Future[Tape...]) = {

  f.onComplete { t: Tape =>
    t.retain()
    try {
      t.backward(t.value)
    } finally {
      t.release()
    }
  }

}
 */
