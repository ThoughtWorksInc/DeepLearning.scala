package com.thoughtworks.deeplearning

import java.util.logging.{Level, LogRecord}

import com.dongxiguo.fastring.Fastring
import com.dongxiguo.fastring.Fastring.Implicits._

object logs {

  private[logs] abstract class LazyLogRecord(level: Level, customMessage: String = null)(
      implicit fullName: sourcecode.FullName,
      methodName: sourcecode.Name,
      className: Caller[_])
      extends LogRecord(level, customMessage) {

    setLoggerName(fullName.value)
    setSourceClassName(className.value.getClass.getName)
    setSourceMethodName(methodName.value)

    protected def makeDefaultMessage: Fastring

    private lazy val defaultMessage: String = makeDefaultMessage.toString

    override def getMessage: String = super.getMessage match {
      case null => defaultMessage
      case message => message
    }

  }

  final case class UncaughtExceptionDuringBackward(
      thrown: Throwable)(implicit fullName: sourcecode.FullName, methodName: sourcecode.Name, className: Caller[_])
      extends LazyLogRecord(Level.SEVERE) {
    setThrown(thrown)
    override protected def makeDefaultMessage = fast"An exception raised during backward"
  }

  final case class DeltaAccumulatorIsUpdating[Delta](
      deltaAccumulator: Delta,
      delta: Delta)(implicit fullName: sourcecode.FullName, methodName: sourcecode.Name, className: Caller[_])
      extends LazyLogRecord(Level.FINER) {
    override protected def makeDefaultMessage: Fastring =
      fast"Before deltaAccumulator update, deltaAccumulator is : $deltaAccumulator, delta is : $delta"
  }

  final case class WeightIsUpdating[Delta](data: Delta, delta: Delta)(implicit fullName: sourcecode.FullName,
                                                                      methodName: sourcecode.Name,
                                                                      className: Caller[_])
      extends LazyLogRecord(Level.FINER) {
    override protected def makeDefaultMessage: Fastring =
      fast"Before weight update, weight is : $data, delta is : $delta"
  }

}
