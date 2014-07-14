package com.qifun.statelessFuture
package util

import org.junit.Test
import org.junit.Assert._
import scala.concurrent.duration._
import java.util.concurrent.Executors
import scala.util.control.Exception.Catcher
/**
 * To make sure it's running Asynchronously
 * */
class TestAsynchronous {

  @Test
  def `HelloWorldTest`() {
    var tmp: Double = 0
    val executor = Executors.newCachedThreadPool()
    @volatile var afterLoop: Long = -2
    @volatile var outOfFuture: Long = -1
    val future = Promise[Unit]
    future.completeWith(Future[Unit] {
      JumpInto(executor).await
      Future {
        for (i <- 1 to 100000000) {
          tmp = scala.math.log10(i.toDouble)
        }
      }.await
      afterLoop = System.currentTimeMillis()
    })
    outOfFuture = System.currentTimeMillis()
      // When `sleep10seconds` is running, it could report failures to this catcher
  implicit def catcher: Catcher[Unit] = {
    case e: Exception => {
      println("An exception occured when I was sleeping: " + e.getMessage)
    }
  }
    (for(unit <- future) assert(afterLoop > outOfFuture))
    Blocking.blockingAwait(future)
  }


  
  @Test
  def `HelloWorldTest2`() {
    val randomDoubleFuture: Future.Stateless[Double] = Future {
      println("Generating a random Double...")
      scala.math.random
    }
    
    val anotherFuture = Future[Unit] {
      println("I am going to read the first random Double.")
      val randomDouble1 = randomDoubleFuture.await
      println(s"The first random Double is $randomDouble1.")
      println("I am going to read the second random Double.")
      val randomDouble2 = randomDoubleFuture.await
      println(s"The second random Double is $randomDouble1.")
    }
    println("Before running the Future.")
      implicit def catcher: Catcher[Unit] = {
    case e: Exception => {
      println("An exception occured when I was sleeping: " + e.getMessage)
    }
  }
    for (unit <- anotherFuture)
    println("After running the Future.")
  }
}