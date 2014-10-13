package com.qifun.statelessFuture.util

import org.junit.Assert._
import org.junit.Test
import com.qifun.statelessFuture.Awaitable

class PipeTest {

  @Test
  def `A FSM should not throw OutOfMemory`(): Unit = {

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

    for (i <- 0 until 100000) {
      writer.write(Event0)
      writer.write(Event1)
      writer.write(Event2)
      writer.write(Event3)
      if (i % 10000 == 0) {
        print(".")
      }
    }

  }

}