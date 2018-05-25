package com.thoughtworks.deeplearning.plugins

import java.nio.ByteBuffer

import com.dongxiguo.fastring.Fastring
import com.dongxiguo.fastring.Fastring.Implicits._
import com.thoughtworks.continuation._
import com.thoughtworks.feature.Caller
import com.typesafe.scalalogging.{CanLog, Logger}
import org.slf4j.MDC
import sourcecode.{FullName, Name}

private object Logging {

  implicit object CanLogSourceCode extends CanLog[(sourcecode.FullName, sourcecode.Name, Caller[_])] {

    private final val mdcKeyFullName = "sourcecode.FullName"
    private final val mdcKeyName = "sourcecode.Name"
    private final val mdcKeyCaller = "com.thoughtworks.feature.Caller"

    def logMessage(originalMessage: String, attachments: (sourcecode.FullName, sourcecode.Name, Caller[_])): String = {

      MDC.put(mdcKeyFullName, attachments._1.value)
      MDC.put(mdcKeyName, attachments._2.value)
      MDC.put(mdcKeyCaller, attachments._3.value.getClass.getCanonicalName)
      originalMessage
    }

    override def afterLog(attachments: (sourcecode.FullName, sourcecode.Name, Caller[_])): Unit = {
      MDC.remove(mdcKeyFullName)
      MDC.remove(mdcKeyName)
      MDC.remove(mdcKeyCaller)
      super.afterLog(attachments)
    }

  }

}

/** A plugin that logs uncaught exceptions.
  *
  * @author 杨博 (Yang Bo)
  */
trait Logging extends Differentiables {
  import Logging._

  protected val logger: Logger

  trait DifferentiableApi extends super.DifferentiableApi {
    implicit protected def fullName: FullName
    implicit protected def name: Name
    implicit protected def caller: Caller[_]
    override protected def handleException(thrown: Throwable): UnitContinuation[Unit] = {
      UnitContinuation.delay {
        Logger
          .takingImplicit[(FullName, Name, Caller[_])](logger.underlying)
          .error("An uncaught exception is thrown", thrown)((fullName, name, caller))
      }
    }
  }

  type Differentiable <: DifferentiableApi

}
