package com.qifun.statelessFuture.util

import org.junit.Assert._
import org.junit.Test
import com.qifun.statelessFuture.Awaitable

class PipeTest {

  @Test
  def `while(true) should not throw OutOfMemory`(): Unit = {

    sealed trait MyEvent
    final case object Event0 extends MyEvent
    final case object Event1 extends MyEvent
    final case object Event2 extends MyEvent
    final case object Event3 extends MyEvent

    val pipe = Pipe[MyEvent]
    def read() = pipe.read()
    val writer = pipe.start(pipe.Future {
      while (true) {
        assertEquals(Event0, read().await)
        assertEquals(Event1, read().await)
        assertEquals(Event2, read().await)
        assertEquals(Event3, read().await)
      }
      throw new IllegalStateException("Unreachable code!")
    })

    for (i <- 0 until 1000000) {
      writer.write(Event0)
      writer.write(Event1)
      writer.write(Event2)
      writer.write(Event3)
    }

  }

  @Test
  def `if / else / match / case should not throw OutOfMemory`(): Unit = {

    sealed trait MyEvent
    final case object Event0 extends MyEvent
    final case object Event1 extends MyEvent
    final case object Event2 extends MyEvent
    final case object Event3 extends MyEvent

    val pipe = Pipe[MyEvent]
    def read() = pipe.read()

    object PingPong {
      def ping(): pipe.Future[Nothing] = pipe.Future[Nothing] {
        assertEquals(Event0, read().await)
        assertEquals(Event1, read().await)
        pong().await
      }

      def pong(): pipe.Future[Nothing] = pipe.Future[Nothing] {
        assertEquals(Event2, read().await)
        assertEquals(Event3, read().await)
        if (math.random < 0.5) {
          math.random match {
            case r if r < 0.5 => {
              ping().await
            }
            case _ => {
              assertEquals(Event0, read().await)
              assertEquals(Event1, read().await)
              pong().await
            }
          }
        } else {
          ping().await
        }
      }
    }

    val writer = pipe.start(PingPong.ping())

    for (i <- 0 until 1000000) {
      writer.write(Event0)
      writer.write(Event1)
      writer.write(Event2)
      writer.write(Event3)
    }

  }

  @Test
  def `ping / pong should not throw OutOfMemory`(): Unit = {

    sealed trait MyEvent
    final case object Event0 extends MyEvent
    final case object Event1 extends MyEvent
    final case object Event2 extends MyEvent
    final case object Event3 extends MyEvent

    val pipe = Pipe[MyEvent]

    object PingPong {
      def ping(): pipe.Future[Nothing] = pipe.Future[Nothing] {
        assertEquals(Event0, pipe.read().await)
        assertEquals(Event1, pipe.read().await)
        pong().await
      }

      def pong(): pipe.Future[Nothing] = pipe.Future[Nothing] {
        assertEquals(Event2, pipe.read().await)
        assertEquals(Event3, pipe.read().await)
        ping().await
      }
    }

    val writer = pipe.start(PingPong.ping())

    for (i <- 0 until 1000000) {
      writer.write(Event0)
      writer.write(Event1)
      writer.write(Event2)
      writer.write(Event3)
    }

  }

}