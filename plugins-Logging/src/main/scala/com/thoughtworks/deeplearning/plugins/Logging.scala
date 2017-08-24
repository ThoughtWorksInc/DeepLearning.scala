package com.thoughtworks.deeplearning.plugins

import java.util.logging.{Level, LogRecord, Logger}

import com.dongxiguo.fastring.Fastring
import com.dongxiguo.fastring.Fastring.Implicits._
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

  final class ThrownInLayer(val layer: Layers#Layer, getThrown: Throwable)(implicit fullName: sourcecode.FullName,
                                                                           name: sourcecode.Name,
                                                                           caller: Caller[_])
      extends ContextualLogRecord(Level.SEVERE, thrown = getThrown)
      with LazyMessage {
    override protected def makeDefaultMessage: Fastring = fast"An exception is thrown in layer $layer"
  }

  final class ThrownInWeight(val weight: Weights#Weight, getThrown: Throwable)(implicit fullName: sourcecode.FullName,
                                                                               name: sourcecode.Name,
                                                                               caller: Caller[_])
      extends ContextualLogRecord(Level.SEVERE, thrown = getThrown)
      with LazyMessage {
    override protected def makeDefaultMessage: Fastring = fast"An exception is thrown in weight $weight"
  }

}

/** A plugin that logs uncaught exceptions raised from [[Layer]] and [[Weight]].
  *
  * @author 杨博 (Yang Bo)
  */
trait Logging extends Layers with Weights {
  import Logging._

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  trait LayerApi extends super.LayerApi { this: Layer =>
    implicit protected def fullName: sourcecode.FullName
    implicit protected def name: sourcecode.Name
    implicit protected def caller: Caller[_]
    override protected def handleException(thrown: Throwable): Unit = {
      logger.log(new ThrownInLayer(this, thrown))
    }
  }
  override type Layer <: LayerApi

  trait WeightApi extends super.WeightApi { this: Weight =>
    implicit protected def fullName: sourcecode.FullName
    implicit protected def name: sourcecode.Name
    implicit protected def caller: Caller[_]
    override protected def handleException(thrown: Throwable): Unit = {
      logger.log(new ThrownInWeight(this, thrown))
    }
  }
  override type Weight <: WeightApi
  override type Implicits <: super[Layers].ImplicitsApi with super[Weights].ImplicitsApi

}
