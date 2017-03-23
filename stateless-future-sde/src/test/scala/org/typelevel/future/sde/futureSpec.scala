package org.typelevel.future.sde

import com.qifun.statelessFuture.Future
import org.scalatest._
import future.AutoImports._
import org.typelevel.future.scalatest.FutureFreeSpec



/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class futureSpec extends FutureFreeSpec with Matchers {

  "try / fireloanally in a sde block should compile" in {
    val f: Future[Int] = future(42)
    future {
      try {
        f.! should be(42)
      } finally {
        f.!
      }
    }
  }

}
