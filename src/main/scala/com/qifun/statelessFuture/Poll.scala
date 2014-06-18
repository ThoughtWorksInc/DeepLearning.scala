package com.qifun.statelessFuture

import java.util.concurrent.atomic.AtomicReference
import scala.util.control.Exception.Catcher
import scala.util.control.TailCalls.TailRec
import scala.util.Try

object Poll {
  /**
   * Returns a [[Poll]] that completes when any one of the `futures` completes,
   * and fails when any one of the `futures` fails,
   */
  def apply[AwaitResult](futures: CancelableFuture[AwaitResult]*) = {
    val result = new Poll[AwaitResult](new AtomicReference(Left(({ () =>
      for (future <- futures) {
        future.cancel()
      }
    }, Nil))))
    for (future <- futures) {
      result.tryCompleteWith(future)
    }
    result
  }
}

/**
 * A [[CancelableFuture]] composed of some sub-futures.
 *
 * This [[CancelableFuture]] completes when any one of the sub-futures completes,
 * and fails when any one of the sub-futures fails,
 */
final class Poll[AwaitResult] private (
  val stateReference: AtomicReference[Either[(() => Unit, List[(AwaitResult => TailRec[Unit], Catcher[TailRec[Unit]])]), Try[AwaitResult]]])
  extends AnyVal with CancelableFuture[AwaitResult]
