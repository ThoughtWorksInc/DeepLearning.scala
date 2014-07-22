package com.qifun.statelessFuture
package util

import scala.concurrent.duration._
import scala.util.control.Exception.Catcher
import com.qifun.statelessFuture.Future

import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit.SECONDS
import org.junit.Test
import org.junit.Assert._
import scala.util.control.TailCalls._



class SleepTest {
  @Test
  def `testSleep`() {
    val executor = Executors.newSingleThreadScheduledExecutor
    val arrayBuffer = scala.collection.mutable.ArrayBuffer(0)
    
    val sleep = Promise[Unit]
    sleep.completeWith(Future[Unit] {
      val sleep1s = Sleep(executor, 1.seconds)
      sleep1s.await
      Future[Unit] {
        println(s"I have slept 1 Seconds.")
        arrayBuffer += 1;
      }.await
      
      Future[Unit] {
        println(s"Another future.")
        arrayBuffer += 2;
      }.await

    })

    assertFalse(sleep.isCompleted)
    println("Before the evaluation of the Stateless Future `sleep`.")
    arrayBuffer += 3;
    Blocking.blockingAwait(sleep)
    assertTrue(sleep.isCompleted)
    assertArrayEquals(arrayBuffer.toArray, Array(0,3,1,2))

  }
  

}