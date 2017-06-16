package com.thoughtworks.deeplearning.scalatest

import scala.concurrent.{Future, Promise}
import scalaz.concurrent.Task
import scalaz.std.`try`
import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo)
  */
trait ScalazTaskToScalaFuture {

  implicit def scalazTaskToScalaFuture[A](task: Task[A]): Future[A] = {
    val promise = Promise[A]
    task.unsafePerformAsync { either =>
      promise.complete(`try`.fromDisjunction(either))
    }
    promise.future
  }

}
