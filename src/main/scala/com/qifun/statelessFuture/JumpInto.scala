package com.qifun.statelessFuture

import scala.util.control.Exception.Catcher
import scala.util.control.TailCalls._
import java.util.concurrent.Executor

/**
 * Let the code after [[JumpInto#await]] run in `executor`.
 * @param executor Where the code after [[JumpInto#await]] run.
 */
final case class JumpInto[TailRecResult](executor: Executor) extends Future.Stateless[Unit] {

  override final def onComplete(handler: Unit => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
    done(executor.execute(new Runnable {
      override final def run(): Unit = {
        handler(()).result
      }
    }))
  }

}
