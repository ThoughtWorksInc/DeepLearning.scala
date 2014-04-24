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
import scala.language.implicitConversions

package object statelessFuture {

  type Future[+AwaitResult] = Awaitable[AwaitResult, Unit]

  object Future {

    type Stateless[+AwaitResult] = Awaitable.Stateless[AwaitResult, Unit]

    type Stateful[+AwaitResult] = Awaitable.Stateful[AwaitResult, Unit]

    def fromResponder[AwaitResult](underlying: Responder[AwaitResult]) = Awaitable.FromResponder(underlying)

    def toResponder[AwaitResult](underlying: Future.Stateless[AwaitResult])(implicit catcher: Catcher[Unit]) = Awaitable.ToResponder(underlying)

    def fromConcurrentFuture[AwaitResult](underlying: scala.concurrent.Future[AwaitResult])(implicit executor: scala.concurrent.ExecutionContext) =
      Awaitable.FromConcurrentFuture(underlying)

    def toConcurrentFuture[AwaitResult](underlying: Future.Stateful[AwaitResult])(implicit intialExecutionContext: ExecutionContext) =
      Awaitable.ToConcurrentFuture(underlying)

    import scala.language.experimental.macros
    def apply[AwaitResult](futureBody: => AwaitResult): Awaitable.Stateless[AwaitResult, Unit] = macro ANormalForm.applyMacro

  }

}