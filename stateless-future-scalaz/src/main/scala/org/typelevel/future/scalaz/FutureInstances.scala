package org.typelevel.future.scalaz

import com.qifun.statelessFuture.Awaitable

import scala.util.control.Exception.Catcher
import scala.util.control.TailCalls.TailRec
import scalaz.MonadError

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class FutureInstances[TailRecResult]
    extends MonadError[({ type T[AwaitResult] = Awaitable[AwaitResult, TailRecResult] })#T, Throwable] {
  override def raiseError[A](e: Throwable) = new Awaitable.Stateless[A, TailRecResult] {
    override def onComplete(handler: (A) => TailRec[TailRecResult])(
        implicit catcher: Catcher[TailRec[TailRecResult]]): TailRec[TailRecResult] = {
      catcher.applyOrElse(e, throw e)
    }
  }

  override def handleError[A](fa: Awaitable[A, TailRecResult])(f: (Throwable) => Awaitable[A, TailRecResult]) =
    new Awaitable.Stateless[A, TailRecResult] {
      override def onComplete(handler: (A) => TailRec[TailRecResult])(
          implicit catcher: Catcher[TailRec[TailRecResult]]): TailRec[TailRecResult] = {
        fa.onComplete { a =>
          handler(a)
        }(PartialFunction { throwable =>
          f(throwable).onComplete(handler)
        })
      }
    }

  override def bind[A, B](fa: Awaitable[A, TailRecResult])(f: (A) => Awaitable[B, TailRecResult]) = {
    fa.flatMap(f)
  }

  override def point[A](a: => A) = new Awaitable.Stateless[A, TailRecResult] {
    override def onComplete(handler: (A) => TailRec[TailRecResult])(
        implicit catcher: Catcher[TailRec[TailRecResult]]): TailRec[TailRecResult] = {
      handler(a)
    }
  }
}

object FutureInstances {

  implicit def futureInstances[TailRecResult]: FutureInstances[TailRecResult] = new FutureInstances[TailRecResult]

}
