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
 * Something that will be completed in the future.
 * @tparam AwaitResult The type that [[await]] returns.
 * @tparam TailRecResult The response type, should be `Unit` in most cases.
 * @see [[Future]]
 */
sealed trait Awaitable[+AwaitResult, TailRecResult] extends Any { outer =>

  /**
   * Suspends this [[Awaitable]] until the asynchronous operation being completed, and then returns the result of the asynchronous operation.
   * @note The code after `await` and the code before `await` may be evaluated in different `Thread`.
   * @note This method must be in a [[Future.apply]] block or [[Awaitable.apply]] block.
   */
  @compileTimeOnly("`await` must be enclosed in a `Future` block")
  final def await: AwaitResult = ???

  /**
   * Like [[foreach]], except this method supports tail-call optimization.
   */
  def onComplete(handler: AwaitResult => TailRec[TailRecResult])(implicit catcher: Catcher[TailRec[TailRecResult]]): TailRec[TailRecResult]


  /**
   * Asks this [[Awaitable]] to pass result to `handler` when the asynchronous operation being completed,
   * or to pass the exception to `catcher` when the asynchronous operation being failed,
   * and starts the asynchronous operation if this [[Awaitable]] is an [[Awaitable.Stateless]]. 
   */
  final def foreach(handler: AwaitResult => TailRecResult)(implicit catcher: Catcher[TailRecResult]): TailRecResult = {
    onComplete { a =>
      done(handler(a))
    } {
      case e if catcher.isDefinedAt(e) =>
        done(catcher(e))
    }.result
  }

  /**
   * Returns a new [[Awaitable.Stateless]] composed of this [[Awaitable]] and the `converter`.
   * The new [[Awaitable.Stateless]] will pass the original result to `convert` when the original asynchronous operation being completed,
   * or pass the exception to `catcher` when the original asynchronous operation being failed.
   */
  final def map[ConvertedAwaitResult](converter: AwaitResult => ConvertedAwaitResult) = new Awaitable.Stateless[ConvertedAwaitResult, TailRecResult] {
    def onComplete(k: ConvertedAwaitResult => TailRec[TailRecResult])(implicit catcher: Catcher[TailRec[TailRecResult]]): TailRec[TailRecResult] = {
      def apply(a: AwaitResult): TailRec[TailRecResult] = {
        val b = try {
          converter(a)
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


  /**
   * Returns a new [[Awaitable.Stateless]] composed of this [[Awaitable]] and the `condition`.
   * The new [[Awaitable.Stateless]] will pass the original result to `condition` when the original asynchronous operation being completed,
   * or pass the exception to `catcher` when the original asynchronous operation being failed.
   * 
   * @throws java.util.NoSuchElementException Passes to `catcher` if the `condition` returns `false`.
   */
  final def withFilter(condition: AwaitResult => Boolean) = new Awaitable.Stateless[AwaitResult, TailRecResult] {
    def onComplete(k: AwaitResult => TailRec[TailRecResult])(implicit catcher: Catcher[TailRec[TailRecResult]]): TailRec[TailRecResult] = {
      def apply(a: AwaitResult): TailRec[TailRecResult] = {
        val b = try {
          condition(a)
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

  /**
   * Returns a new [[Awaitable.Stateless]] composed of this [[Awaitable]] and the `converter`.
   * The new [[Awaitable.Stateless]] will pass the original result to `convert` when the original asynchronous operation being completed,
   * or pass the exception to `catcher` when the original asynchronous operation being failed.
   */
  final def flatMap[ConvertedAwaitResult](converter: AwaitResult => Awaitable[ConvertedAwaitResult, TailRecResult]) = new Awaitable.Stateless[ConvertedAwaitResult, TailRecResult] {
    override final def onComplete(body: ConvertedAwaitResult => TailRec[TailRecResult])(implicit catcher: Catcher[TailRec[TailRecResult]]): TailRec[TailRecResult] = {
      def apply(a: AwaitResult): TailRec[TailRecResult] = {
        val futureB = try {
          converter(a)
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

  /**
   * An stateful [[Awaitable]] that represents an asynchronous operation already started.
   */
  trait Stateful[+AwaitResult, TailRecResult] extends Any with Awaitable[AwaitResult, TailRecResult] {

    /**
     * Tests if this [[Awaitable.Stateful]] is completed.
     */
    def isCompleted: Boolean = value.isDefined

    /**
     * The result of the asynchronous operation.
     */
    def value: Option[Try[AwaitResult]]

  }

  /**
   * An stateless [[Awaitable]] that starts a new asynchronous operation whenever [[onComplete]] or [[foreach]] being called.
   * @note The result value of the operation will never store in this [[Stateless]].
   */
  trait Stateless[+AwaitResult, TailRecResult] extends Any with Awaitable[AwaitResult, TailRecResult]

  import scala.language.experimental.macros
  
  /**
   * Returns a stateless [[Awaitable]] that evaluates the `block`.
   * @param block The asynchronous operation that will perform later. Note that all [[Awaitable.await]] calls must be in the `block`. 
   */
  def apply[AwaitResult, TailRecResult](block: => AwaitResult): Awaitable.Stateless[AwaitResult, TailRecResult] = macro ANormalForm.applyMacro

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