package com.qifun.statelessFuture

import org.junit.Test
import org.junit.Assert._
import com.qifun.statelessFuture.AwaitableSeq._

class AwaitableSeqTest {
  @Test
  def `simple yield`() {
    val l = List(1, 2, 3)
    val f = Future {
      for (i <- futureSeq(l)) yield {
        val sb = Seq.newBuilder[Int]
        sb += i
        Future {
          sb += 99
        }.await
        sb += i
        sb.result
      }
    }
    implicit def catcher = PartialFunction.empty
    f.foreach { seq =>
      assertEquals(
        Seq(Seq(1, 99, 1), Seq(2, 99, 2), Seq(3, 99, 3)),
        seq.underlying)
    }
  }

  @Test
  def `complex yield`() {
    val l1 = List(1, 2, 3)
    val l2 = Vector(4, 5, 6)
    val l3 = Array(7, 8)
    val f = Future {
      for {
        i <- futureSeq(l1)
        if i != 2
        j <- futureSeq(l2)
        d = j + i
        k <- futureSeq(l3)
      } yield {
        val sb = Seq.newBuilder[Int]
        sb += i
        Future {
          sb += 99
        }.await
        sb += d
        sb += k
        sb.result
      }
    }
    implicit def catcher = PartialFunction.empty
    f.foreach { seq =>
      assertEquals(
        Seq(Seq(1, 99, 5, 7), Seq(1, 99, 5, 8), Seq(1, 99, 6, 7), Seq(1, 99, 6, 8), Seq(1, 99, 7, 7), Seq(1, 99, 7, 8), Seq(3, 99, 7, 7), Seq(3, 99, 7, 8), Seq(3, 99, 8, 7), Seq(3, 99, 8, 8), Seq(3, 99, 9, 7), Seq(3, 99, 9, 8)),
        seq.underlying)
    }

  }

  @Test
  def `simple list comprehension`() {
    val sb = Seq.newBuilder[Int]
    val l = List(1, 2, 3)
    val f = Future {
      for (i <- futureSeq(l)) {
        sb += i
        Future {
          sb += 99
        }.await
        sb += i
      }
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
      for {
        i <- futureSeq(l1)
        if i != 2
        j <- futureSeq(l2)
        k <- futureSeq(l3)
      } {
        sb += i
        Future {
          sb += 99
        }.await
        sb += j
        sb += k
      }
    }
    implicit def catcher = PartialFunction.empty
    f.foreach { u =>
      assertEquals(sb.result, Seq(1, 99, 4, 7, 1, 99, 4, 8, 1, 99, 5, 7, 1, 99, 5, 8, 1, 99, 6, 7, 1, 99, 6, 8, 3, 99, 4, 7, 3, 99, 4, 8, 3, 99, 5, 7, 3, 99, 5, 8, 3, 99, 6, 7, 3, 99, 6, 8))
    }

  }
}