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

package com.qifun

import scala.util._
import scala.util.control.TailCalls._
import scala.util.control.Exception.Catcher
import scala.concurrent.ExecutionContext

package object statelessFuture {

  type Future[+AwaitResult] = Awaitable[AwaitResult, Unit]

  object Future {

    type Stateless[+AwaitResult] = Awaitable.Stateless[AwaitResult, Unit]

    type Stateful[+AwaitResult] = Awaitable.Stateful[AwaitResult, Unit]

    final class FromConcurrentFuture[AwaitResult](underlying: scala.concurrent.Future[AwaitResult])(implicit executor: scala.concurrent.ExecutionContext) extends Future.Stateful[AwaitResult] {
      import scala.util._

      override final def value = underlying.value

      override final def onComplete(body: AwaitResult => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
        underlying.onComplete {
          case Success(successValue) => {
            executor.execute(new Runnable {
              override final def run(): Unit = {
                body(successValue).result
              }
            })
          }
          case Failure(throwable) => {
            if (catcher.isDefinedAt(throwable)) {
              executor.execute(new Runnable {
                override final def run(): Unit = {
                  catcher(throwable).result
                }
              })
            } else {
              executor.prepare.reportFailure(throwable)
            }
          }
        }
        done(())
      }

    }

    final class ToConcurrentFuture[AwaitResult](underlying: Future.Stateful[AwaitResult])(implicit intialExecutionContext: ExecutionContext) extends scala.concurrent.Future[AwaitResult] {

      import scala.concurrent._
      import scala.concurrent.duration.Duration
      import scala.util._

      override final def value: Option[Try[AwaitResult]] = underlying.value

      override final def isCompleted = underlying.isCompleted

      override final def onComplete[U](func: (Try[AwaitResult]) => U)(implicit executor: ExecutionContext) {
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

      override final def result(atMost: Duration)(implicit permit: CanAwait): AwaitResult = {
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

    final class FromResponder[AwaitResult](underlying: Responder[AwaitResult]) extends Future.Stateless[AwaitResult] {
      override def onComplete(body: AwaitResult => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
        try {
          underlying.respond { a =>
            body(a).result
          }
        } catch {
          case e if catcher.isDefinedAt(e) =>
            catcher(e).result
        }
        done(())
      }
    }

    final class ToResponder[AwaitResult](underlying: Future.Stateless[AwaitResult])(implicit catcher: Catcher[Unit]) extends Responder[AwaitResult] {

      override final def respond(handler: AwaitResult => Unit) {
        (underlying.onComplete { a =>
          done(handler(a))
        } {
          case e if catcher.isDefinedAt(e) => {
            done(catcher(e))
          }
        }).result
      }

    }

    import scala.language.experimental.macros
    def apply[AwaitResult](futureBody: => AwaitResult): Awaitable.Stateless[AwaitResult, Unit] = macro ANormalForm.applyMacro

  }

}