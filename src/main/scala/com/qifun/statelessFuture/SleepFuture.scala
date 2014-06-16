package com.qifun.statelessFuture

import java.util.concurrent.ScheduledExecutorService
import scala.concurrent.duration.Duration
import scala.util.control.Exception.Catcher
import scala.util.control.TailCalls._
import scala.util.control.TailCalls
import scala.util.Success
import scala.util.Failure
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.ScheduledFuture
import scala.util.Try
import java.util.concurrent.CancellationException

final class SleepFuture private (
  val state: AtomicReference[Either[(() => Unit, List[(Unit => TailRec[Unit], Catcher[TailRec[Unit]])]), Try[Unit]]])
  extends AnyVal with CancelableFuture[Unit]

object SleepFuture {
  def apply(executor: ScheduledExecutorService, duration: Duration): SleepFuture = {
    if (duration.isFinite) {
      object UnderlyingRunnable extends Runnable {
        val underlyingFuture = executor.schedule(this, duration.length, duration.unit)

        val result = new SleepFuture(new AtomicReference(Left((
          { () => underlyingFuture.cancel(false) },
          Nil))))
        override final def run() {
          result.tryComplete(Success(()))
        }
      }
      UnderlyingRunnable.result
    } else {
      new SleepFuture(new AtomicReference(Left((
        { () => },
        Nil))))
    }
  }
}