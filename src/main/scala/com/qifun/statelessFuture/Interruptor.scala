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

import java.io.Closeable
import scala.util.control.TailCalls._
import scala.util.control.Exception.Catcher
import scala.util.Success
import scala.util.Try

object Interruptor {

  class FutureInterruptedException(message: String = null, cause: Throwable = null) extends Exception(message, cause)

  private final class Poll[A] private[Interruptor] (onInterrupt: Future[Nothing], interruptableFutureFactories: Seq[Interruptor => Future[A]]) extends StatelessFuture[A] {

    override final def onComplete(body: A => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
      val promise = Promise[A]()
      val innerInterrupter = new Interruptor {
        override final val onInterrupt = Future {
          promise.await
          throw new FutureInterruptedException("Cancel all the polling futures because one of them have been completed.")
        }
      }
      val i = interruptableFutureFactories.iterator
      def loop(): TailRec[Unit] = {
        if (i.hasNext) {
          val future = i.next()(innerInterrupter)
          promise.tryCompleteWith(future).flatMap { u => loop() }
        } else {
          done(())
        }
      }
      loop().flatMap { u =>
        promise.tryCompleteWith(onInterrupt)
      }.flatMap { u =>
        promise.onComplete(body)(catcher)
      }
    }

  }

  import scala.language.implicitConversions

  implicit def interruptable[A <: Future[_]](futureFactory: Interruptor => A)(implicit interruptor: Interruptor) =
    futureFactory(interruptor)

}

trait Interruptor {

  def onInterrupt: Future[Nothing]

  def poll[A](interruptableFutureFactories: (Interruptor => Future[A])*): StatelessFuture[A] =
    new Interruptor.Poll(onInterrupt, interruptableFutureFactories)

}