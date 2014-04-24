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

import scala.reflect.internal.annotations.compileTimeOnly
import scala.util.control.TailCalls._
import scala.util.control.Exception.Catcher
import scala.util.Try
import scala.concurrent.ExecutionContext
import scala.util.Success
import scala.util.Failure

sealed trait Awaitable[+AwaitResult, TailRecResult] extends Any { outer =>

  @compileTimeOnly("`await` must be enclosed in a `Future` block")
  final def await: AwaitResult = ???

  def onComplete(body: AwaitResult => TailRec[TailRecResult])(implicit catcher: Catcher[TailRec[TailRecResult]]): TailRec[TailRecResult]

  final def foreach(f: AwaitResult => TailRecResult)(implicit catcher: Catcher[TailRecResult]): TailRecResult = {
    onComplete { a =>
      done(f(a))
    } {
      case e if catcher.isDefinedAt(e) =>
        done(catcher(e))
    }.result
  }

  final def map[B](f: AwaitResult => B) = new Awaitable.Stateless[B, TailRecResult] {
    def onComplete(k: B => TailRec[TailRecResult])(implicit catcher: Catcher[TailRec[TailRecResult]]): TailRec[TailRecResult] = {
      def apply(a: AwaitResult): TailRec[TailRecResult] = {
        val b = try {
          f(a)
        } catch {
          case e if catcher.isDefinedAt(e) => {
            return tailcall(catcher(e))
          }
        }
        tailcall(k(b))
      }
      outer.onComplete(apply)
    }
  }

  final def withFilter(p: AwaitResult => Boolean) = new Awaitable.Stateless[AwaitResult, TailRecResult] {
    def onComplete(k: AwaitResult => TailRec[TailRecResult])(implicit catcher: Catcher[TailRec[TailRecResult]]): TailRec[TailRecResult] = {
      def apply(a: AwaitResult): TailRec[TailRecResult] = {
        val b = try {
          p(a)
        } catch {
          case e if catcher.isDefinedAt(e) => {
            return tailcall(catcher(e))
          }
        }
        if (b) {
          tailcall(k(a))
        } else {
          tailcall(catcher(new NoSuchElementException))
        }
      }
      outer.onComplete(apply)
    }
  }

  final def flatMap[B](mapping: AwaitResult => Awaitable[B, TailRecResult]) = new Awaitable.Stateless[B, TailRecResult] {
    override final def onComplete(body: B => TailRec[TailRecResult])(implicit catcher: Catcher[TailRec[TailRecResult]]): TailRec[TailRecResult] = {
      def apply(a: AwaitResult): TailRec[TailRecResult] = {
        val futureB = try {
          mapping(a)
        } catch {
          case e if catcher.isDefinedAt(e) => {
            return tailcall(catcher(e))
          }
        }
        futureB.onComplete { b =>
          tailcall(body(b))
        }
      }
      outer.onComplete(apply)
    }
  }

}

object Awaitable {

  trait Stateful[+AwaitResult, TailRecResult] extends Any with Awaitable[AwaitResult, TailRecResult] {

    def isCompleted: Boolean = value.isDefined

    def value: Option[Try[AwaitResult]]

  }

  trait Stateless[+AwaitResult, TailRecResult] extends Any with Awaitable[AwaitResult, TailRecResult]

  implicit final class FromConcurrentFuture[AwaitResult](underlying: scala.concurrent.Future[AwaitResult])(implicit executor: scala.concurrent.ExecutionContext) extends Future.Stateful[AwaitResult] {
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

  implicit final class ToConcurrentFuture[AwaitResult](underlying: Future.Stateful[AwaitResult])(implicit intialExecutionContext: ExecutionContext) extends scala.concurrent.Future[AwaitResult] {

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

  implicit final class FromResponder[AwaitResult](underlying: Responder[AwaitResult]) extends Future.Stateless[AwaitResult] {
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

  implicit final class ToResponder[AwaitResult](underlying: Future.Stateless[AwaitResult])(implicit catcher: Catcher[Unit]) extends Responder[AwaitResult] {

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
  def apply[AwaitResult, TailRecResult](futureBody: => AwaitResult): Awaitable.Stateless[AwaitResult, TailRecResult] = macro ANormalForm.applyMacro

}