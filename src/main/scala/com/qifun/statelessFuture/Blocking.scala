package com.qifun.statelessFuture

import scala.util.Success
import scala.util.Try
import scala.util.Failure
import scala.util.control.Exception.Catcher

object Blocking {

  final def blockingAwait[A](future: Future[A]): A = {
    val lock = new AnyRef
    lock.synchronized {
      @volatile var result: Option[Try[A]] = None
      implicit def catcher: Catcher[Unit] = {
        case e: Exception => {
          lock.synchronized {
            result = Some(Failure(e))
            lock.notifyAll()
          }
        }
      }
      future.foreach { u =>
        lock.synchronized {
          result = Some(Success(u))
          lock.notify()
        }
      }
      while (result == None) {
        lock.wait()
      }
      val Some(some) = result
      some match {
        case Success(u) => u
        case Failure(e) => throw e
      }
    }
  }
}