package com.qifun.statelessFuture
package util

import scala.concurrent.duration.Duration
import java.util.concurrent.ScheduledExecutorService
import scala.util.Success

object Sleep {

  def apply(executor: ScheduledExecutorService, duration: Duration): CancelablePromise[Unit] = {
    if (duration.isFinite) {
      object UnderlyingRunnable extends Runnable {
        val underlyingFuture = executor.schedule(this, duration.length, duration.unit)
        val result = CancelablePromise[Unit] { () => underlyingFuture.cancel(false) }
        override final def run() {
          result.tryComplete(Success(()))
        }
      }
      UnderlyingRunnable.result
    } else {
      CancelablePromise { () => }
    }
  }

}