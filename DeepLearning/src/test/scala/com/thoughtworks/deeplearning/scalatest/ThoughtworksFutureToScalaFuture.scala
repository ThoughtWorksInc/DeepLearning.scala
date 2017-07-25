package com.thoughtworks.deeplearning.scalatest

import scala.language.implicitConversions
import com.thoughtworks.future._

/**
  * @author 杨博 (Yang Bo)
  */
trait ThoughtworksFutureToScalaFuture {

  implicit def thoughtworksFutureToScalaFuture[A](future: Future[A]): scala.concurrent.Future[A] = {
    future.toScalaFuture
  }

}
