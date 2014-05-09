package com.qifun.statelessFuture
import org.junit._
import scala.annotation.tailrec
import Assert._
class GeneratorTest {

  @Test
  def `generate "Hello, World!"`() {
    implicit val gen = Generator[Char]
    val seq: gen.OutputSeq = gen.Future {
      'H'.await
      gen("ello": _*).await
      gen(',', ' ').await
      val world = "World!".iterator
      while (world.hasNext) {
        world.next().await
      }
    }
    assertEquals(seq.mkString, "Hello, World!")
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
        wrap(seq: gen.Future[Unit], layer - 1)
      }
    }

    val wrapped100 = wrap(future1('first, 'last), 100)
    assertEquals(wrapped100(0), 'first)
    assertEquals(wrapped100(50), 'dynamic49)
    assertEquals(wrapped100(204), 'last)

  }

}

// vim: set ts=2 sw=2 et:
