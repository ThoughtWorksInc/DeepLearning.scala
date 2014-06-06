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
import org.junit._
import scala.annotation.tailrec
import Assert._
import com.qifun.statelessFuture.AwaitableSeq._
import java.io.File
class GeneratorTest {

  @Test
  def `foreach with a diffent type`() {
    val sourceDirectoriesValue: Seq[File] = Seq(new File("xxxxxx"), new File("foobar"))
    val gen = Generator[String]
    val seq: gen.OutputSeq = gen.Future[Unit] {
      gen.futureSeq(sourceDirectoriesValue).foreach { d =>
        gen("-I", d.getPath).await
      }
      gen("--cache-dir", "xx").await
    }
    assertEquals(Seq("-I", "xxxxxx", "-I", "foobar", "--cache-dir", "xx"), seq)
  }

  @Test
  def `generate "Hello, World!"`() {
    implicit val gen = Generator[Char]
    val seq: gen.OutputSeq = gen.Future {
      'H'.await
      gen("ello": _*).await
      gen(',', ' ').await
      val world = "Wo".iterator
      while (world.hasNext) {
        world.next().await
      }
      for (c <- gen.futureSeq("rld!")) {
        gen(c).await
      }
    }
    assertEquals("Hello, World!", seq.mkString)
  }

  @Test
  def `generate Any elements`() {
    implicit val gen = Generator[Any]
    val seq: gen.OutputSeq = gen.Future {
      // gen(1).await // Does not compile due to https://issues.scala-lang.org/browse/SI-2991
      (gen: (Any => gen.Future[Unit]))(1).await
    }
    assertEquals(Seq(1), seq)
  }

  @Test
  def `convert between seq and future`() {
    implicit val gen = Generator[Symbol]
    def future1(first: Symbol, last: Symbol) = gen.Future {
      first.await
      val symbols = for (i <- 0 until 200) yield Symbol(raw"dynamic$i")
      gen(symbols: _*).await
      'foo.await
      'bar.await
      'baz.await
      last.await
    }

    @tailrec
    def wrap(seq: gen.OutputSeq, layer: Int): gen.OutputSeq = {
      if (layer == 0) {
        seq
      } else {
        wrap(Generator.Seq(seq), layer - 1)
      }
    }

    val wrapped100 = wrap(future1('first, 'last), 100)
    assertEquals(wrapped100(0), 'first)
    assertEquals(wrapped100(50), 'dynamic49)
    assertEquals(wrapped100(204), 'last)

  }

}

// vim: set ts=2 sw=2 et:
