package org.typelevel.future.scalatest

import com.qifun.statelessFuture.Future
import com.qifun.statelessFuture.util.Promise
import org.scalactic.source
import org.scalatest._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait FutureFreeSpec extends AsyncFreeSpec {

  protected implicit final class StringInOps(message: String)(implicit pos: source.Position) {

    def in(statelessFuture: Future[compatible.Assertion]): Unit = {
      val p = Promise[compatible.Assertion]
      p.completeWith(statelessFuture)
      convertToFreeSpecStringWrapper(message).in(p: scala.concurrent.Future[compatible.Assertion])
    }

  }
}
