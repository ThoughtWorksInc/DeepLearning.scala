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

import org.junit.Test
import org.junit.Assert._
import scala.concurrent.duration._
import java.util.concurrent.Executors
import scala.util.control.Exception.Catcher

object TestAsynchronous {
  private implicit val (logger, formatter, appender) = ZeroLoggerFactory.newLogger(this)
}

/**
 * To make sure it's running Asynchronously
 */
final class TestAsynchronous {
  import TestAsynchronous._
  @Test
  def `HelloWorldTest`() {
    @volatile var tmp: Double = 0
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
        assertFalse(outOfFuture == -1)
      }.await
      afterLoop = System.currentTimeMillis()
    })
    outOfFuture = System.currentTimeMillis()
    // When `sleep10seconds` is running, it could report failures to this catcher
    implicit def catcher: Catcher[Unit] = {
      case e: Exception => {
        logger.warning("An exception occured when I was sleeping: " + e.getMessage)
      }
    }
    for (unit <- future) assertTrue(afterLoop > outOfFuture)
    Blocking.blockingAwait(future)
  }

  @Test
  def `HelloWorldTest2`() {

    var randomFutureCalledCounter: Int = 0;

    val randomDoubleFuture: Future.Stateless[Double] = Future {
      logger.fine("Generating a random Double...")
      randomFutureCalledCounter += 1
      scala.math.random
    }

    val anotherFuture = Future[Unit] {
      logger.fine("I am going to read the first random Double.")
      assertEquals(randomFutureCalledCounter, 0)
      val randomDouble1 = randomDoubleFuture.await
      assertEquals(randomFutureCalledCounter, 1)
      logger.fine(s"The first random Double is $randomDouble1.")
      logger.fine("I am going to read the second random Double.")
      val randomDouble2 = randomDoubleFuture.await
      assertEquals(randomFutureCalledCounter, 2)
      logger.fine(s"The second random Double is $randomDouble1.")
    }
    assertTrue(randomFutureCalledCounter < 2)
    logger.fine("Before running the Future.")
    implicit def catcher: Catcher[Unit] = {
      case e: Exception => {
        logger.warning("An exception occured when I was sleeping: " + e.getMessage)
      }
    }
    Blocking.blockingAwait(anotherFuture)
    logger.fine("After running the Future.")
  }
}