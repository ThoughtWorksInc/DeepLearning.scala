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
    var future1State: Boolean = false
    var future2State: Boolean = false
    var future3State: Boolean = false

    val myCancellableFuture1 = CancellablePromise[Unit]

    myCancellableFuture1.completeWith[Unit](Future {
      JumpInto(executor).await
      println("start future1")
      Future {
        println("start for in future1")
        for (i <- 1 to 500000) counter1 = i
        future1State = true
        println("future1 hass been completed")
      }.await
    }).result

    val myCancellableFuture2 = CancellablePromise[Unit]

    myCancellableFuture2.completeWith[Unit](Future {
      JumpInto(executor).await
      println("start future2")
      Future {
        println("start for in future2")
        for (i <- 1 to 1000000) counter2 = i
        future2State = true
      }.await
      println("future2 hass been completed")
    }).result

    val myCancellableFuture3 = CancellablePromise[Unit]

    myCancellableFuture3.completeWith[Unit](Future {
      JumpInto(executor).await
      println("start future3")
      Future {
        println("start for in future3")
        for (i <- 1 to 2000000) counter2 = i
        future3State = true
      }.await
      println("future3 hass been completed")
    }).result

    val myPoll = Poll(myCancellableFuture1, myCancellableFuture2, myCancellableFuture3)

    implicit def catcher: Catcher[Unit] = {
      case e: Exception => {
        println("An exception occured when I was sleeping: " + e.getMessage)
      }
    }
    var hasFinished = false
    for (result <- myPoll) {
      println("Poll return a result")
      println(s"the state of futures: 1:$future1State 2:$future2State 3:$future3State")
      assert(future1State && (!future2State) && (!future3State))
      hasFinished = true
    }
    Blocking.blockingAwait(myPoll)
    assert(hasFinished)
    println("afterBlockingAwait for Poll")
    Blocking.blockingAwait(myCancellableFuture2)
    assert(future2State)
    Blocking.blockingAwait(myCancellableFuture3)
    assert(future3State)
  }
}
