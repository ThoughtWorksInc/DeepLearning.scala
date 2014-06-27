package com.qifun.statelessFuture.util


object Poll {

  /**
   * Returns a [[Poll]] that completes when any one of the `futures` completes,
   * and fails when any one of the `futures` fails,
   */
  def apply[AwaitResult](futures: CancelableFuture[AwaitResult]*) = {
    val result = CancelablePromise[AwaitResult] { () =>
      for (future <- futures) {
        future.cancel()
      }
    }
    for (future <- futures) {
      result.tryCompleteWith(future)
    }
    result
  }
}