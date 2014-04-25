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

/**
 * An asynchronous operation that will be completed in the future.
 * @tparam AwaitResult The type that [[#await]] returns.
 * @tparam TailRecResult The response type, should be `Unit` in most cases.
 * @see [[Future]]
 */
sealed trait Awaitable[+AwaitResult, TailRecResult] extends Any { outer =>

  /**
   * Suspends this [[Awaitable]] until the asynchronous operation complete, and then returns the result of the asynchronous operation.
   * @note The code after `await` and the code before `await` may be evaluated in different `Thread`.
   */
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

  import scala.language.experimental.macros
  def apply[AwaitResult, TailRecResult](futureBody: => AwaitResult): Awaitable.Stateless[AwaitResult, TailRecResult] = macro ANormalForm.applyMacro

  import scala.language.implicitConversions

  implicit def fromResponder[AwaitResult](underlying: Responder[AwaitResult]) = {
    new Future.FromResponder(underlying)
  }

  implicit def toResponder[AwaitResult](underlying: Future.Stateless[AwaitResult])(implicit catcher: Catcher[Unit]) = {
    new Future.ToResponder(underlying)
  }

  implicit def fromConcurrentFuture[AwaitResult](underlying: scala.concurrent.Future[AwaitResult])(implicit executor: scala.concurrent.ExecutionContext) = {
    new Future.FromConcurrentFuture(underlying)
  }

  implicit def toConcurrentFuture[AwaitResult](underlying: Future.Stateful[AwaitResult])(implicit intialExecutionContext: ExecutionContext) = {
    new Future.ToConcurrentFuture(underlying)
  }

}