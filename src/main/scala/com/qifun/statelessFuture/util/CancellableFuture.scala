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
import scala.util.Try
import scala.util.control.TailCalls._
import scala.util.control.Exception.Catcher

/**
 * A [[Future.Stateful]] that will be completed when another [[Future]] being completed.
 * @param stateReference The internal stateReference that should never be accessed by other modules.
 */
trait CancellableFuture[AwaitResult] extends Any with Future.Stateful[AwaitResult] {

  def cancel(): Unit

  final def cancelWith(implicit cancellationToken: CancellationToken) = {
    val registration = cancellationToken.register(() => cancel())
    foreach(_ => cancellationToken.unregister(registration)) {
      case _ => cancellationToken.unregister(registration)
    }
    this
  }
}