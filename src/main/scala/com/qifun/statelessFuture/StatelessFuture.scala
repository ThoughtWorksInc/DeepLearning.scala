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

import scala.runtime.AbstractPartialFunction
import scala.reflect.macros.Context
import scala.util.control.Exception.Catcher
import scala.util.control.TailCalls._
import scala.concurrent.ExecutionContext
import scala.util.Success
import scala.util.Failure

/**
 * @author 杨博
 */
trait StatelessFuture[+A] extends Any with Future[A]

object StatelessFuture {

  import scala.language.experimental.macros
  def apply[A](futureBody: => A): StatelessFuture[A] = macro ANormalForm.applyMacro

  import scala.language.implicitConversions
  implicit def toConcurrentFuture[A](underlying: StatelessFuture[A])(implicit intialExecutionContext: ExecutionContext): scala.concurrent.Future[A] = {
    val p = Promise[A]()
    intialExecutionContext.execute(new Runnable {
      override final def run(): Unit = {
        p.completeWith(underlying).result
      }
    })
    StatefulFuture.ToConcurrentFuture(p)
  }
}
