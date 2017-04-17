package com.thoughtworks.deeplearning

import org.slf4j.LoggerFactory

trait Logger {
  def handleBackwardException(throwable: Throwable)
  def handleForwardException(throwable: Throwable)
}

object Logger {

  private val logger: org.slf4j.Logger = LoggerFactory.getLogger(classOf[Logger])

  implicit def defaultLogger: Logger = {
    new Logger {
      override def handleBackwardException(throwable: Throwable): Unit =
        logger.error("An exception raised during backward", throwable)

      override def handleForwardException(throwable: Throwable): Unit =
        logger.error("An exception raised during forward", throwable)
    }
  }
}
