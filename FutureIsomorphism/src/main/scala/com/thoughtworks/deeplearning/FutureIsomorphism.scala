package com.thoughtworks.deeplearning

import scalaz.{ContT, Trampoline}
import scalaz.Free.Trampoline
import scalaz.concurrent.Future

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object FutureIsomorphism extends scalaz.Isomorphism.IsoFunctorTemplate[Future, ContT[Trampoline, Unit, ?]] {
  override def to[A](fa: Future[A]): ContT[Trampoline, Unit, A] = ContT[Trampoline, Unit, A] { continue =>
    Trampoline.delay(fa.unsafePerformListen(continue))
  }

  override def from[A](ga: ContT[Trampoline, Unit, A]): Future[A] = {
    Future.Async { continue =>
      ga(continue).run
    }
  }
}
