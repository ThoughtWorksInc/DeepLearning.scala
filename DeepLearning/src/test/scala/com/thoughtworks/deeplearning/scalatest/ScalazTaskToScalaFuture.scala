package com.thoughtworks.deeplearning.scalatest

import scala.concurrent.Promise
import scala.language.implicitConversions
import scalaz.Trampoline

/**
  * @author 杨博 (Yang Bo)
  */
trait ScalazTaskToScalaFuture {

  implicit def scalazTaskToScalaFuture[A](task: com.thoughtworks.future.Future[A]): scala.concurrent.Future[A] = {
    val promise = Promise[A]
    com.thoughtworks.future.Future
      .run(task) { either =>
        Trampoline.delay {
          val _ = promise.complete(either)
        }
      }
      .run
    promise.future
  }

}
