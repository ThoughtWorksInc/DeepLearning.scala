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
import com.qifun.statelessFuture.util.AwaitableSeq._

final class AwaitableSeqTest {
  @Test
  def `simple for/yield`() {
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
  def `complex for/yield`() {
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
  def `simple for`() {
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
  def `complex for`() {
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