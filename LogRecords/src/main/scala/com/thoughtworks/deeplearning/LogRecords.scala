package com.thoughtworks.deeplearning

import java.util.logging.{Level, LogRecord}

import com.dongxiguo.fastring.Fastring
import com.dongxiguo.fastring.Fastring.Implicits._

object LogRecords {

  private[LogRecords] abstract class LazyLogRecord(level: Level, customMessage: String = null)(
      implicit fullName: sourcecode.FullName,
      methodName: sourcecode.Name,
      fileName: sourcecode.File)
      extends LogRecord(level, customMessage) {

    setLoggerName(fullName.value)
    setSourceClassName(fileName.value)
    setSourceMethodName(methodName.value)

    protected def makeDefaultMessage: Fastring

    private lazy val defaultMessage: String = makeDefaultMessage.toString

    override def getMessage: String = super.getMessage match {
      case null => defaultMessage
      case message => message
    }

  }

  final case class UncaughtExceptionDuringBackward(thrown: Throwable) extends LazyLogRecord(Level.SEVERE) {
    setThrown(thrown)
    override protected def makeDefaultMessage = fast"An exception raised during backward"
  }

  final case class DeltaAccumulatorTracker[Delta](deltaAccumulator: Delta, delta: Delta)
      extends LazyLogRecord(Level.FINER) {
    override protected def makeDefaultMessage: Fastring = fast"deltaAccumulator : $deltaAccumulator, delta : $delta"
  }

  final case class FloatWeightTracker[Delta](data: Delta, delta: Delta) extends LazyLogRecord(Level.FINER) {
    override protected def makeDefaultMessage: Fastring = fast"weight : $data, delta : $delta"
  }

}
