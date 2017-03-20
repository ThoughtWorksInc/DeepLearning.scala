package com.qifun.statelessFuture.util

import com.qifun.statelessFuture.Awaitable

import scala.util.{Success, Try}
import scala.util.control.Exception.Catcher
import scala.util.control.TailCalls.TailRec

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class Constant[+AwaitResult, TailRecResult](result: AwaitResult)
    extends Awaitable.Stateful[AwaitResult, TailRecResult] {

  override def isCompleted: Boolean = true

  override def value: Option[Try[AwaitResult]] = Some(Success(result))

  override def onComplete(handler: (AwaitResult) => TailRec[TailRecResult])(
      implicit catcher: Catcher[TailRec[TailRecResult]]): TailRec[TailRecResult] = {
    handler(result)
  }
}
