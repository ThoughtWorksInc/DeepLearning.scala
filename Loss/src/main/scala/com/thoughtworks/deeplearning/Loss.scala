package com.thoughtworks.deeplearning


import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import com.thoughtworks.raii.covariant.ResourceT
import com.thoughtworks.raii.covariant.ResourceT._

import scalaz.concurrent.Task
import scalaz.std.`try`.toDisjunction
import scalaz.syntax.all._

trait Loss[-Data, +Delta] {
  /** Returns the delta of loss for given `scores` */
  def deltaLoss(`scores`: Data): Do[Delta]
}

object Loss {
  def train[OutputData, OutputDelta](forward: Do[Tape[OutputData, OutputDelta]])(
      implicit trainable: Loss[OutputData, OutputDelta]): Task[OutputData] = {

    val doOutputData: Do[OutputData] = forward.flatMap[OutputData] { outputTape =>
      Do.delay(outputTape.backward(trainable.deltaLoss(outputTape.data))).map { _ =>
        outputTape.data
      }
    }

    val Do(futureOutputData) = doOutputData
    new Task(ResourceT.run(ResourceT(futureOutputData).map(toDisjunction)))
  }
}
