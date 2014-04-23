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

import scala.util.control.TailCalls._
import scala.util.control.Exception.Catcher
import scala.concurrent.ExecutionContext
import scala.reflect.internal.annotations.compileTimeOnly

trait Future[+A] extends Any { outer =>

  @compileTimeOnly("`await` must be enclosed in a `Future` block")
  final def await: A = ???

  def onComplete(body: A => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit]

  final def foreach[U](f: A => U)(implicit catcher: Catcher[Unit]) {
    onComplete { a =>
      f(a)
      done(())
    } {
      case e if catcher.isDefinedAt(e) =>
        catcher(e)
        done(())
    }.result
  }

  final def map[B](f: A => B) = new StatelessFuture[B] {
    def onComplete(k: B => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
      def apply(a: A): TailRec[Unit] = {
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

  final def withFilter(p: A => Boolean) = new StatelessFuture[A] {
    def onComplete(k: A => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
      def apply(a: A): TailRec[Unit] = {
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

  final def flatMap[B](mapping: A => Future[B]) = new StatelessFuture[B] {
    override final def onComplete(body: B => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
      def apply(a: A): TailRec[Unit] = {
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

object Future {

  implicit final class FromConcurrentFuture[A](underlying: scala.concurrent.Future[A])(implicit executor: scala.concurrent.ExecutionContext) extends StatefulFuture[A] {
    import scala.util._

    override final def value = underlying.value

    override final def onComplete(body: A => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
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

  import scala.language.implicitConversions
  implicit def toConcurrentFuture[A](underlying: Future[A])(implicit intialExecutionContext: ExecutionContext): scala.concurrent.Future[A] = {
    underlying match {
      case statelessFuture: StatelessFuture[A] => StatelessFuture.toConcurrentFuture(statelessFuture)
      case statefulFuture: StatefulFuture[A] => new StatefulFuture.ToConcurrentFuture(statefulFuture)
    }
  }

  implicit final class FromResponder[A](underlying: Responder[A]) extends Future[A] {
    override def onComplete(body: A => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
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

  implicit final class ToResponder[A](underlying: Future[A])(implicit catcher: Catcher[Unit]) extends Responder[A] {

    override final def respond(handler: A => Unit) {
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
  def apply[A](futureBody: => A): StatelessFuture[A] = macro ANormalForm.applyMacro
}
