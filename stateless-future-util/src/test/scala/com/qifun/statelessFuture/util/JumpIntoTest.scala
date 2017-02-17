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

import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit.SECONDS
import org.junit.Test
import scala.util.control.TailCalls._
import scala.concurrent.duration._
import org.junit.Assert._
import scala.util.control.Exception.Catcher
import java.util.concurrent.ThreadFactory
import scala.collection.mutable.ListBuffer
import AwaitableSeq._

final class JumpIntoTest {

  @Test
  def `jump into test`() {
    val executor = Executors.newFixedThreadPool(5, new ThreadFactory {
      override final def newThread(r: Runnable): Thread = {
        new Thread(r, "executor's thread")
      }
    })
    Thread.currentThread.setName("main thread")
    val ji10times = Future {
      for (i <- futureSeq(0 until 10)) {
        if (i == 0) {
          assertEquals("main thread", Thread.currentThread.getName)
        }
        JumpInto(executor).await
        assertEquals("executor's thread", Thread.currentThread.getName)
      }
    }
    Blocking.blockingAwait(ji10times)
    executor.shutdown()
  }
}
