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
package io

import java.nio.channels._
import org.junit.Assert._
import org.junit.Test
import java.util.concurrent.Executors
import scala.util.control.Exception.Catcher
import scala.util.Try
import scala.util.Failure
import scala.util.Success
import java.net.InetAddress
import java.net.InetSocketAddress
import scala.concurrent.duration._
import java.nio.ByteBuffer

class SocketTest {

  private def blockingAwait[A](future: Future[A]): A = {
    val lock = new AnyRef
    lock.synchronized {
      @volatile var result: Option[Try[A]] = None
      implicit def catcher: Catcher[Unit] = {
        case e: Exception => {
          lock.synchronized {
            result = Some(Failure(e))
            lock.notifyAll()
          }
        }
      }
      future.foreach { u =>
        lock.synchronized {
          result = Some(Success(u))
          lock.notify()
        }
      }
      while (result == None) {
        lock.wait()
      }
      val Some(some) = result
      some match {
        case Success(u) => u
        case Failure(e) => throw e
      }
    }
  }

  @Test
  def pingPong() {
    val executor = Executors.newSingleThreadScheduledExecutor
    try {
      val channelGroup = AsynchronousChannelGroup.withThreadPool(executor)
      try {
        val serverSocket = AsynchronousServerSocketChannel.open(channelGroup)
        try {
          serverSocket.bind(null)
          object MyException extends Exception
          val clientFuture = Future[Unit] {
            val socket0 = AsynchronousSocketChannel.open()
            try {
              Nio2Future.connect(socket0, new InetSocketAddress("localhost", serverSocket.getLocalAddress.asInstanceOf[InetSocketAddress].getPort)).await
              val stream0 = new SocketInputStream with SocketWritingQueue {
                val socket = socket0
                def readingTimeout = Duration.Inf
                def writingTimeout = 3.seconds
              }
              stream0.enqueue(ByteBuffer.wrap("ping".getBytes("UTF-8")))
              stream0.flush()
              stream0.available_=(4).await
              assertEquals(4, stream0.available)
              val pong = Array.ofDim[Byte](4)
              val numBytesRead = stream0.read(pong)
              assertEquals(4, numBytesRead)
              assertEquals("pong", new String(pong, "UTF-8"))
              stream0.interrupt()
              throw MyException
            } finally {
              socket0.close()
            }
          }
          val serverFuture = Future {
            val socket1 = Nio2Future.accept(serverSocket).await
            val stream1 = new SocketInputStream with SocketWritingQueue {
              val socket = socket1
              def readingTimeout = Duration.Inf
              def writingTimeout = 3.seconds
            }
            try {
              stream1.available_=(4).await
              val ping = Array.ofDim[Byte](4)
              val numBytesRead = stream1.read(ping)
              assertEquals(4, numBytesRead)
              assertEquals("ping", new String(ping, "UTF-8"))
              stream1.enqueue(ByteBuffer.wrap("pong".getBytes("UTF-8")))
              stream1.shutDown()
            } finally {
              socket1.close()
            }
          }
          try {
            blockingAwait(Zip(Promise.completeWith(clientFuture), Promise.completeWith(serverFuture)))
            throw new AssertionError("Expect MyException.")
          } catch {
            case MyException =>
          }
        } finally {
          serverSocket.close()
        }
      } finally {
        channelGroup.shutdown()
      }
    } finally {
      executor.shutdownNow()
    }

  }
}