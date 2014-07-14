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

package com.qifun.statelessFuture.util

import scala.util.control.Exception.Catcher
import scala.util.control.TailCalls
import java.util.concurrent.CancellationException

object Poll {

  /**
   * Returns a [[Poll]] that completes when any one of the `futures` completes,
   * and fails when any one of the `futures` fails.
   *
   * When a [[Poll]] completes, it cancels all [[futures]].
   */
  def apply[AwaitResult](futures: CancellableFuture[AwaitResult]*): Poll[AwaitResult] = {
    val result = CancellablePromise[AwaitResult]
    val onCancel: Catcher[Unit] = {
      case _: CancellationException =>
        for (future <- futures) {
          future.cancel()
        }
    }
    (for (_ <- result) {
      // success
    })(onCancel)
    for (future <- futures) {
      result.tryCompleteWith(future)
    }
    result
  }
}