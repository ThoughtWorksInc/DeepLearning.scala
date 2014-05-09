package com.qifun.statelessFuture

import org.junit.Test
import org.junit.Assert._
import AwaitableSeq._

class AwaitableSeqTest {
  @Test
  def `simple yield`() {
    val l = List(1, 2, 3)
    val f = Future {
      (for (i <- l.asFutureSeq) yield Future {

        val sb = Seq.newBuilder[Int]
        sb += i
        Future {
          sb += 99
        }.await
        sb += i
        sb.result
      }).await
    }
    implicit def catcher = PartialFunction.empty
    f.foreach { seq =>
      assertEquals(seq, Seq(Seq(1, 99, 1), Seq(2, 99, 2), Seq(3, 99, 3)))
    }
  }

  @Test
  def `complex yield`() {
    val l1 = List(1, 2, 3)
    val l2 = Vector(4, 5, 6)
    val l3 = Array(7, 8)
    val f = Future {
      (for {
        i <- l1.asFutureSeq
        if i != 2
        j <- l2.asFutureSeq
        k <- l3.asFutureSeq
      } yield Future {
        val sb = Seq.newBuilder[Int]
        sb += i
        Future {
          sb += 99
        }.await
        sb += j
        sb += k
        sb.result
      }).await
    }
    implicit def catcher = PartialFunction.empty
    f.foreach { seq =>
      assertEquals(seq, Seq(Seq(1, 99, 4, 7), Seq(1, 99, 4, 8), Seq(1, 99, 5, 7), Seq(1, 99, 5, 8), Seq(1, 99, 6, 7), Seq(1, 99, 6, 8), Seq(3, 99, 4, 7), Seq(3, 99, 4, 8), Seq(3, 99, 5, 7), Seq(3, 99, 5, 8), Seq(3, 99, 6, 7), Seq(3, 99, 6, 8)))
    }

  }

  @Test
  def `simple list comprehension`() {
    val sb = Seq.newBuilder[Int]
    val l = List(1, 2, 3)
    val f = Future {
      (for (i <- l.asFutureSeq) Future {
        sb += i
        Future {
          sb += 99
        }.await
        sb += i
      }).await
    }
    implicit def catcher = PartialFunction.empty
    f.foreach { u =>
      assertEquals(sb.result, Seq(1, 99, 1, 2, 99, 2, 3, 99, 3))
    }
  }

  @Test
  def `complex list comprehension`() {
    val sb = Seq.newBuilder[Int]
    val l1 = List(1, 2, 3)
    val l2 = Vector(4, 5, 6)
    val l3 = Array(7, 8)
    val f = Future {
      (for {
        i <- l1.asFutureSeq
        if i != 2
        j <- l2.asFutureSeq
        k <- l3.asFutureSeq
      } Future {
        sb += i
        Future {
          sb += 99
        }.await
        sb += j
        sb += k
      }).await
    }
    implicit def catcher = PartialFunction.empty
    f.foreach { u =>
      assertEquals(sb.result, Seq(1, 99, 4, 7, 1, 99, 4, 8, 1, 99, 5, 7, 1, 99, 5, 8, 1, 99, 6, 7, 1, 99, 6, 8, 3, 99, 4, 7, 3, 99, 4, 8, 3, 99, 5, 7, 3, 99, 5, 8, 3, 99, 6, 7, 3, 99, 6, 8))
    }

  }
}