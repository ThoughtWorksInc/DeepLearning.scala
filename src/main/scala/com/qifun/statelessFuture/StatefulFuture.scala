/*
 * stateless-future
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

import scala.util.Try
import scala.util.control.TailCalls._
import scala.util.control.Exception.Catcher
import scala.concurrent.ExecutionContext

trait StatefulFuture[+A] extends Any with Future[A] {

  def isCompleted: Boolean = value.isDefined

  def value: Option[Try[A]]

}

object StatefulFuture {

  implicit final class ToConcurrentFuture[A](underlying: StatefulFuture[A])(implicit intialExecutionContext: ExecutionContext) extends scala.concurrent.Future[A] {

    import scala.concurrent._
    import scala.concurrent.duration.Duration
    import scala.util._

    override final def value: Option[Try[A]] = underlying.value

    override final def isCompleted = underlying.isCompleted

    override final def onComplete[U](func: (Try[A]) => U)(implicit executor: ExecutionContext) {
      underlying.onComplete { a =>
        executor.prepare.execute(new Runnable {
          override final def run() {
            ToConcurrentFuture.this.synchronized {
              try {
                func(Success(a))
              } catch {
                case e: Throwable => executor.reportFailure(e)
              }
            }
          }
        })
        done(())

      } {
        case e: Throwable => {
          executor.prepare.execute(new Runnable {
            override final def run() {
              ToConcurrentFuture.this.synchronized {
                try {
                  func(Failure(e))
                } catch {
                  case e: Throwable => executor.reportFailure(e)
                }
              }
            }
          })
          done(())
        }
      }

    }

    override final def result(atMost: Duration)(implicit permit: CanAwait): A = {
      ready(atMost)
      value.get match {
        case Success(successValue) => successValue
        case Failure(throwable) => throw throwable
      }
    }

    override final def ready(atMost: Duration)(implicit permit: CanAwait): this.type = {
      if (atMost eq Duration.Undefined) {
        throw new IllegalArgumentException
      }
      synchronized {
        if (atMost.isFinite) {
          val timeoutAt = atMost.toNanos + System.nanoTime
          while (!isCompleted) {
            val restDuration = timeoutAt - System.nanoTime
            if (restDuration < 0) {
              throw new TimeoutException
            }
            wait(restDuration / 1000000, (restDuration % 1000000).toInt)
          }
        } else {
          while (!isCompleted) {
            wait()
          }
        }
        this
      }
    }

    implicit private def catcher: Catcher[Unit] = {
      case throwable: Throwable => {
        synchronized {
          notifyAll()
        }
      }
    }

    for (successValue <- underlying) {
      synchronized {
        notifyAll()
      }
    }
  }
}