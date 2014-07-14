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
    new AnyValCancellablePromise[AwaitResult](new Underlying[AwaitResult](Left(Queue.empty)))

  final class AnyValCancellablePromise[AwaitResult] private[CancellablePromise] (
    val state: AtomicReference[Either[Queue[(AwaitResult => TailRec[Unit], Catcher[TailRec[Unit]])], Try[AwaitResult]]])
    extends AnyVal with CancellablePromise[AwaitResult]

}

/**
 * A [[Future.Stateful]] that will be completed when another [[Future]] being completed.
 *
 * @param stateReference The internal stateReference that should never be accessed by other modules.
 */
trait CancellablePromise[AwaitResult]
  extends Any with Promise[AwaitResult] with CancellableFuture[AwaitResult] {

  // 提供类似C#的隐式参数CancellationToken，对用户来说比较易用。
  // 提供CancelableFuture，对实现者来说，更自然，因为更贴近Java底层的API。而且也避免了一处事件回调。
  // 如果做人工智能，总是需要自己实现类似CancellationToken的机制。比如我先前做的Interruptor.

  final def cancel() {
    tryComplete(Failure(new CancellationException)).result
  }

}