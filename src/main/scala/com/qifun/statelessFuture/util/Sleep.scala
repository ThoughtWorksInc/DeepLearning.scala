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

import scala.concurrent.duration.Duration
import java.util.concurrent.ScheduledExecutorService
import scala.util.Success
import java.util.concurrent.CancellationException
import scala.util.control.Exception.Catcher
import java.util.concurrent.ScheduledFuture

object Sleep {

  def start(promise: Promise[Unit], executor: ScheduledExecutorService, duration: Duration): Unit = {
    if (duration.isFinite) {
      val _ = new Runnable {

        override final def run() {
          promise.tryComplete(Success(())).result
        }

        /**
         * @note 此处 startTimer()有副作用，是为了避免把underlyingFuture设为var
         */ 
        private val underlyingFuture = executor.schedule(this, duration.length, duration.unit)

        (for (_ <- promise) {
          // success
        }) {
          case _: CancellationException =>
            val _ = underlyingFuture.cancel(false)
        }

      }
    }
  }

  def apply(executor: ScheduledExecutorService, duration: Duration): Sleep = {
    val result = CancellablePromise[Unit]
    start(result, executor, duration)
    result
  }

}
