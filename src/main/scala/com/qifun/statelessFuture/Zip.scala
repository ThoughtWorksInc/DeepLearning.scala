package com.qifun.statelessFuture

import scala.util.Success
import scala.util.Failure
import scala.util.control.Exception.Catcher
import scala.util.control.TailCalls._

final case class Zip[ThisAwaitResult, ThatAwaitResult](
  thisFuture: Future.Stateful[ThisAwaitResult],
  thatFuture: Future.Stateful[ThatAwaitResult]) extends Future.Stateful[(ThisAwaitResult, ThatAwaitResult)] {

  override final def value = {
    (thisFuture.value, thatFuture.value) match {
      case (Some(Success(thisSuccess)), Some(Success(thatSuccess))) => Some(Success((thisSuccess, thatSuccess)))
      case (Some(Failure(e)), _) => Some(Failure(e))
      case (_, Some(Failure(e))) => Some(Failure(e))
      case _ => None
    }
  } 

  override final def onComplete(handler: ((ThisAwaitResult, ThatAwaitResult)) => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
    thisFuture.onComplete { thisSuccess =>
      thatFuture.onComplete { thatSuccess =>
        handler((thisSuccess, thatSuccess))
      }
    }
  }

}
