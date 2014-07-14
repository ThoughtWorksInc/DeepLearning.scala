package com.qifun.statelessFuture
package util

import java.util.concurrent.Executors
import org.junit.Test
import org.junit.Assert._

class ZipTest {
  @Test
  def `zipTest1`() {
    @volatile var future1Counter: Int = 0;
    @volatile var future2Counter: Int = 0;
    val executor = Executors.newCachedThreadPool()

    val myFuture: Future.Stateless[Unit] = Future[Unit] {

      
      val future1 = Promise[Unit]
      future1.completeWith(Future[Unit] {
        JumpInto(executor).await
        Future {
          println(System.currentTimeMillis())
          for (i <- 1 to 500000) {
            future1Counter = i;
          }
        }.await
      })

      
      val future2 = Promise[Unit]
      future2.completeWith(Future[Unit] {
        JumpInto(executor).await
        Future {
          println(System.currentTimeMillis())
          for (i <- 1 to 800000) {
            future2Counter = i;
          }
        }.await
      })

      val futureZip = new Zip(future1, future2)
      futureZip.await
      println("FutureZip process successful")
    }

    Blocking.blockingAwait(myFuture)

    println(s"after FutureZip await, future1Counter = $future1Counter, future2Counter = $future2Counter")
    assertEquals(future1Counter, 500000)
    assertEquals(future2Counter, 800000)
  }
}