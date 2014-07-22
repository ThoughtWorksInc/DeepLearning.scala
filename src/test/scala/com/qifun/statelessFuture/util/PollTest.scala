package com.qifun.statelessFuture
package util

import java.util.concurrent.Executors
import org.junit.Test
import org.junit.Assert._
import scala.util.control.Exception.Catcher

class PollTest {
  @Test
  def `pollTest1`() {
    val executor = Executors.newCachedThreadPool()
    @volatile var counter1: Int = 0
    @volatile var counter2: Int = 0
    @volatile var counter3: Int = 0
    @volatile var isFuture1Finished: Boolean = false
    @volatile var isFuture2Finished: Boolean = false
    @volatile var isFuture3Finished: Boolean = false

    val myCancellableFuture1 = CancellablePromise[Unit]

    myCancellableFuture1.completeWith[Unit](Future {
      JumpInto(executor).await
      println("start future1")
      Future {
        println("start for in future1")
        for (i <- 1 to 500000) counter1 = i
        isFuture1Finished = true
        println("future1 hass been completed")
      }.await
    })

    val myCancellableFuture2 = CancellablePromise[Unit]

    myCancellableFuture2.completeWith[Unit](Future {
      JumpInto(executor).await
      println("start future2")
      Future {
        println("start for in future2")
        for (i <- 1 to 1000000) counter2 = i
        isFuture2Finished = true
      }.await
      println("future2 hass been completed")
    })

    val myCancellableFuture3 = CancellablePromise[Unit]

    myCancellableFuture3.completeWith[Unit](Future {
      JumpInto(executor).await
      println("start future3")
      Future {
        println("start for in future3")
        for (i <- 1 to 2000000) counter2 = i
        isFuture3Finished = true
      }.await
      println("future3 hass been completed")
    })

    val myPoll = Poll(myCancellableFuture1, myCancellableFuture2, myCancellableFuture3)

    implicit def catcher: Catcher[Unit] = {
      case e: Exception => {
        println("An exception occured when I was sleeping: " + e.getMessage)
      }
    }
    var hasFinished = false
    for (result <- myPoll) {
      println("Poll return a result")
      println(s"the state of futures: 1:isFuture1Finished 2:isFuture2Finished 3:isFuture3Finished")
      assertTrue(isFuture1Finished)
      assertFalse(isFuture2Finished)
      assertFalse(isFuture3Finished)
      hasFinished = true
    }
    Blocking.blockingAwait(myPoll)
    assertTrue(hasFinished)
    println("afterBlockingAwait for Poll")
    Blocking.blockingAwait(myCancellableFuture2)
    assertTrue(isFuture2Finished)
    Blocking.blockingAwait(myCancellableFuture3)
    assertTrue(isFuture3Finished)
  }
}
