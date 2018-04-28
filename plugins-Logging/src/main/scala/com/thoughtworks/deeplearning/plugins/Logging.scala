package com.thoughtworks.deeplearning.plugins

import java.util.logging.{Level, LogRecord, Logger}

import com.dongxiguo.fastring.Fastring
import com.dongxiguo.fastring.Fastring.Implicits._
import com.thoughtworks.continuation._
import com.thoughtworks.feature.Caller

object Logging {

  /** A [[java.util.logging.LogRecord LogRecord]] that contains current source contextual information. */
  class ContextualLogRecord(
      level: Level,
      message: String = null,
      parameters: Array[AnyRef] = null,
      thrown: Throwable = null)(implicit fullName: sourcecode.FullName, name: sourcecode.Name, caller: Caller[_])
      extends LogRecord(level, message) {

    setParameters(parameters)
    setThrown(thrown)
    setLoggerName(fullName.value)
    setSourceClassName(caller.value.getClass.getName)
    setSourceMethodName(name.value)
  }

  trait LazyMessage extends LogRecord {
    protected def makeDefaultMessage: Fastring

    private lazy val defaultMessage: String = makeDefaultMessage.toString

    override def getMessage: String = super.getMessage match {
      case null    => defaultMessage
      case message => message
    }
  }
  final class UncaughtException(val differentiable: Logging#DifferentiableApi, getThrown: Throwable)(
      implicit fullName: sourcecode.FullName,
      name: sourcecode.Name,
      caller: Caller[_])
      extends ContextualLogRecord(Level.SEVERE, thrown = getThrown)
      with LazyMessage {
    override protected def makeDefaultMessage: Fastring = fast"An exception is thrown in $differentiable"
  }

}

/** A plugin that logs uncaught exceptions.
  *
  * @author 杨博 (Yang Bo)
  */
trait Logging extends Differentiables {
  import Logging._

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  trait DifferentiableApi extends super.DifferentiableApi {
    implicit protected def fullName: sourcecode.FullName
    implicit protected def name: sourcecode.Name
    implicit protected def caller: Caller[_]
    override protected def handleException(thrown: Throwable): UnitContinuation[Unit] = {
      UnitContinuation.delay {
        logger.log(new UncaughtException(this, thrown))
      }
    }
  }

  type Differentiable <: DifferentiableApi

}
