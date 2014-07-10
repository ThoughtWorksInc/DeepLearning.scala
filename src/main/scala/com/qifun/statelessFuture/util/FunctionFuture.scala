package com.qifun.statelessFuture
package util

import scala.util.control.TailCalls._
import scala.util.control.Exception.Catcher

/**
 * A simple future implemented by [[foreachFunction]].
 *
 * @param foreachFunction The function that provides implementation for [[foreach]].
 */
final class FunctionFuture[AwaitResult](val foreachFunction: (AwaitResult => Unit, Catcher[Unit]) => Unit)
  extends AnyVal with Future.Stateless[AwaitResult] {

  override final def onComplete(
    handler: AwaitResult => TailRec[Unit])(
      implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
    done(foreachFunction(handler.andThen(_.result), catcher.andThen(_.result)))
  }

}