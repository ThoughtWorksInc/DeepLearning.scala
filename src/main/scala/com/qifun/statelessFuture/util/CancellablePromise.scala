/*
 * stateless-future-util
 * Copyright 2014 深圳岂凡网络有限公司 (Shenzhen QiFun Network Corp., LTD)
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.qifun.statelessFuture
package util

import java.util.concurrent.atomic.AtomicReference
import scala.util.control.Exception.Catcher
import scala.util.control.TailCalls._
import scala.util.Failure
import java.util.concurrent.CancellationException
import scala.util.Success
import scala.util.Try
import scala.collection.immutable.Queue

object CancellablePromise {

  private implicit class Scala210TailRec[A](underlying: TailRec[A]) {
    final def flatMap[B](f: A => TailRec[B]): TailRec[B] = {
      tailcall(f(underlying.result))
    }
  }

  private type CancelFunction = () => Unit

  private type HandlerList[AwaitResult] = Queue[(AwaitResult => TailRec[Unit], Catcher[TailRec[Unit]])]

  private type State[AwaitResult] = Either[HandlerList[AwaitResult], Try[AwaitResult]]

  private type Underlying[AwaitResult] = AtomicReference[State[AwaitResult]]

  final def apply[AwaitResult] =
    new CancellablePromise[AwaitResult](new Underlying[AwaitResult](Left(Queue.empty)))

}

/**
 * A [[Future.Stateful]] that will be completed when another [[Future]] being completed.
 *
 * @param stateReference The internal stateReference that should never be accessed by other modules.
 */
final class CancellablePromise[AwaitResult] private (
  // TODO: 把List和Tuple2合并成一个对象，以减少内存占用
  val stateReference: CancellablePromise.Underlying[AwaitResult])
  extends AnyVal with CancellableFuture[AwaitResult] {

  // 提供类似C#的隐式参数CancellationToken，对用户来说比较易用。
  // 提供CancelableFuture，对实现者来说，更自然，因为更贴近Java底层的API。而且也避免了一处事件回调。
  // 如果做人工智能，总是需要自己实现类似CancellationToken的机制。比如我先前做的Interruptor.

  final def cancel() {
    stateReference.get match {
      case oldState @ Left(handlers) => {
        val value = Failure(new CancellationException)
        if (stateReference.compareAndSet(oldState, Right(value))) {
          tailcall(dispatch(handlers, value))
        } else {
          cancel()
        }
      }
      case _ => {
        // Ignore
      }
    }
  }

  private def dispatch(
    handlers: Queue[(AwaitResult => TailRec[Unit], Catcher[TailRec[Unit]])], value: Try[AwaitResult]): TailRec[Unit] = {
    // 为了能在Scala 2.10中编译通过
    import CancellablePromise.Scala210TailRec
    handlers.dequeueOption match {
      case None => {
        done(())
      }
      case Some(((body, catcher), tail)) => {
        (value match {
          case Success(a) => {
            body(a)
          }
          case Failure(e) => {
            if (catcher.isDefinedAt(e)) {
              catcher(e)
            } else {
              done(())
            }
          }
        }).flatMap { _ =>
          dispatch(tail, value)
        }
      }
    }
  }

  override final def value = stateReference.get.right.toOption

  // @tailrec // Comment this because of https://issues.scala-lang.org/browse/SI-6574
  final def complete(value: Try[AwaitResult]): TailRec[Unit] = {
    stateReference.get match {
      case oldState @ Left(handlers) => {
        if (stateReference.compareAndSet(oldState, Right(value))) {
          tailcall(dispatch(handlers, value))
        } else {
          complete(value)
        }
      }
      case Right(origin) => {
        throw new IllegalStateException("Cannot complete a CancellablePromise twice!")
      }
    }
  }

  /**
   * Starts a waiting operation that will be completed when `other` being completed.
   * @throws java.lang.IllegalStateException when this [[CancellablePromise]] is completed more once.
   * @usecase def completeWith(other: Future[AwaitResult]): TailRec[Unit] = ???
   */
  final def completeWith[OriginalAwaitResult](
    other: Future[OriginalAwaitResult])(
      implicit view: OriginalAwaitResult => AwaitResult): TailRec[Unit] = {
    other.onComplete { b =>
      val value = Success(view(b))
      tailcall(complete(value))
    } {
      case e: Throwable => {
        val value = Failure(e)
        tailcall(complete(value))
      }
    }
  }

  // @tailrec // Comment this annotation because of https://issues.scala-lang.org/browse/SI-6574
  final def tryComplete(value: Try[AwaitResult]): TailRec[Unit] = {
    stateReference.get match {
      case oldState @ Left(handlers) => {
        if (stateReference.compareAndSet(oldState, Right(value))) {
          tailcall(dispatch(handlers, value))
        } else {
          tryComplete(value)
        }
      }
      case Right(origin) => {
        done(())
      }
    }
  }

  /**
   * Starts a waiting operation that will be completed when `other` being completed.
   * Unlike [[completeWith]], no exception will be created when this [[CancellablePromise]] being completed more once.
   * @usecase def tryCompleteWith(other: Future[AwaitResult]): TailRec[Unit] = ???
   */
  final def tryCompleteWith[OriginalAwaitResult](
    other: Future[OriginalAwaitResult])(
      implicit view: OriginalAwaitResult => AwaitResult): TailRec[Unit] = {
    other.onComplete { b =>
      val value = Success(view(b))
      tailcall(tryComplete(value))
    } {
      case e: Throwable => {
        val value = Failure(e)
        tailcall(tryComplete(value))
      }
    }
  }

  // @tailrec // Comment this annotation because of https://issues.scala-lang.org/browse/SI-6574
  override final def onComplete(
    body: AwaitResult => TailRec[Unit])(
      implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
    stateReference.get match {
      case Right(value) => {
        value match {
          case Success(a) => {
            body(a)
          }
          case Failure(e) => {
            if (catcher.isDefinedAt(e)) {
              catcher(e)
            } else {
              throw e
            }
          }
        }
      }
      case oldState @ Left(tail) => {
        if (stateReference.compareAndSet(oldState, Left(tail.enqueue((body, catcher))))) {
          done(())
        } else {
          onComplete(body)
        }
      }
    }
  }

}