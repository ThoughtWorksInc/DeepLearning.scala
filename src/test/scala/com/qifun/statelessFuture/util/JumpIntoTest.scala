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

class JumpIntoTest {

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
